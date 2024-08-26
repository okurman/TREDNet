# -*- coding: utf-8 -*-
"""
@author: okurman
"""

from kipoi.model import KerasModel


class PhaseOneModel(KerasModel):
    def __init__(self, weights, arch):
        super().__init__(weights=weights, arch=arch)
