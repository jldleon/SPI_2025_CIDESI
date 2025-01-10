from __future__ import annotations
from typing import Literal

import torch, re
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sentence_transformers import SentenceTransformer

# For the test
import requests 

class CMM_MobileNet:
    def __init__(self) -> None:
        self.preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
        self.classifier = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
        self.class2id:dict[str:int] = self.classifier.config.label2id
        self._cosine_sim = nn.CosineSimilarity( dim=0 )
        # For LM filtering
        self._mini_lm = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
        # For debbuging purposes
        # logtis, phi, m, w_a, c_a 
        self.calc_vals:dict[str:object] = {}

    # Base filters
    base_filters = {
        "all" : lambda k_ij, clss: True,
        "substring" : lambda k_ij, clss: k_ij in clss,
        "subword" : lambda k_ij, clss: re.compile(f".*(^|\s){k_ij}($|\s|,).*").match(clss) is not None
        }
    
    def _semantic_filter(self, kwd:str, classes:list[str], n:int|float) -> list[str]:
        """Selects the most similar words from classes to kwd, according to n (int) higher scores or those above a threshold n(float)"""
        embds = self._mini_lm.encode( [kwd] + classes, convert_to_tensor=True)
        embds_scores = self._mini_lm.similarity( embds[0][None,:], embds[1:] )[0]
        embds_scores = sorted( zip( embds_scores, range( len(embds_scores) ) ), reverse=True)
        if isinstance(n,int):
            filtered_classes = [ classes[i] for _, i in embds_scores[:n] ]
        else:
            filtered_classes = [ classes[i] for score, i in embds_scores if score>=n]
        return filtered_classes

    def phi_map(self, k:str, filter_fn:function=None, semantic_selection:int|float=0, background_class:bool=False ) -> list[list[int]]:
        """
        Returns a list of all the classes for all the keywords.
            Args:
                - k: list of keywords
                - selection_fn: Function that says if a keyword belongs to a class given: selection_fn(keyword, class) -> bool
                                If no function given no filtering will be done but exact fit of keywords in class dictionary.
                - semantic_selection: Tells if semantic class selection will be used in the result of selection_fn, which uses the 
                                        'paraphrase-MiniLM-L6-v2' model to compare the similarity. If int, higher n class are selected,
                                        if float, all classes with a n similarity score or higher.
                - background_class: When True, empty keyword positions [] will be filled with 0 idx.
        """
        # If no filter => the keyword itself is a class => sematic selection has no sense
        assert semantic_selection==False or filter_fn is not None, "If 'semantic_selection' is needed, classes must be selected by 'filter_fn'"

        if filter_fn is None:
            phi_i = [ [ kwd ] for kwd in k ]
        else:
            phi_i = [ [ class_k for class_k in self.class2id.keys() if filter_fn( kwd, class_k ) ]
                     for kwd in k ]
            
        if semantic_selection!=False:
            phi_i = [ self._semantic_filter(kwd_j, phi_ij, semantic_selection )
                     for kwd_j, phi_ij in zip(k,phi_i) ]
        
        self.calc_vals["phi_class"] = phi_i

        phi_i = [ [ self.class2id[clss] for clss in phi_ij ] if phi_ij!=[] else
                    ( [0] if background_class else [] )
                    for phi_ij in phi_i ]
        self.calc_vals["phi"] = phi_i
        return phi_i

    def _calculate_adhesions(self, kw:dict[str:int], logits:torch.Tensor, *phi_args, **phi_kwargs ) -> tuple[torch.Tensor]:
        """
        Calculates Weight and Class adhesion
            - Args:
                - kw: keyword-weight dictionary
                - logits: direct result of classifier model on generated image to evaluate
        """
        k_i, w_i = zip(*kw.items())
        phi_i = self.phi_map( k_i, *phi_args, **phi_kwargs )
        probs_i = torch.softmax( logits, dim=0 )
        m_i = torch.Tensor([ sum([logits[i].item() for i in phii_j ]) for phii_j in phi_i ])
        weight_adhesion = self._cosine_sim( torch.Tensor(w_i), m_i )
        class_adhesion = sum([ sum([probs_i[k] for k in phi_ij ]) for phi_ij in phi_i ])
        self.calc_vals["m"] = m_i
        self.calc_vals["w_a"] = weight_adhesion
        self.calc_vals["c_a"] = class_adhesion
        return (weight_adhesion, class_adhesion)

    def calculate(self, kw:dict[str:int], image:Image.Image,
                metric_variant:Literal["multiplicative", "similitude", "average", "all"]="average",
                filter_fn:Literal["all","substring","subword"]|function=None,
                semantic_selection:int|float=0, background_class:bool=False ) -> torch.Tensor|tuple[torch.Tensor]:
        """
        Calculates Class Measure Metric.
            - Args:
                - kw: keyword-weight dictionary
                - image: image to be evaluated
                - metric_variant: Which of the variants of CMM will be returned.
                - filter_fn: Function to select the class(es) for each keyword. Predefined functions are:
                       - all: All classes are acepted.
                       - substring: Checks if keyword is substring of class.
                       - subword: Impose that keyword is a complete and stand alone word in class.
                       If None (default), metric consider that keywords are exact classes from imagenet.
                - semantic_selection: If semantic selection is used to reduced classes when filtered.
                                   If int n classes are selected, when float, classes above threshold.
                - background_class: If there's no fitting class for keyword then a 0 idx will be assign.
            
            - Return: If 'metric_variant'!='both' metric value as tensor, otherwhise, tuple with both values.
        """
        assert metric_variant in ["multiplicative", "similitude", "average", "all"], "Bad metric variant"

        inputs = self.preprocessor(images=image, return_tensors="pt")
        outputs = self.classifier(**inputs)
        logits = outputs.logits.detach()[0]
        self.calc_vals["logits"] = logits
        filter_fn = CMM_MobileNet.base_filters[filter_fn] if type(filter_fn)==str else filter_fn
        w_a, c_a = self._calculate_adhesions( kw, logits, filter_fn , semantic_selection, background_class )
        metric = {
            "multiplicative" : w_a*c_a, # mcmm
            "similitude" : (w_a+c_a)/torch.sqrt( 2*( w_a**2 + c_a**2 ) ), # scmm
            "average" : (w_a+c_a)/2, # acmm
        }
        metric["all"] = tuple(metric.values())
        return metric[metric_variant]
    

if __name__=="__main__":
    # Intended to test correct construction of CMM_MobileNet object
    # and to give an example of execution
    cmm = CMM_MobileNet()
    url_wolf = "https://images.fineartamerica.com/images-medium-large-5/grey-wolves-playing-william-ervinscience-photo-library.jpg"
    img_wolf = Image.open(requests.get(url_wolf, stream=True).raw)
    kw = {"wolf":100, "snow":20}
    mcmm, scmm, acmm = cmm.calculate( kw, img_wolf, "all", "all", True, False )
    print( "mcmm: ", mcmm )
    print( "scmm: ", scmm )
    print( "acmm: ", acmm )