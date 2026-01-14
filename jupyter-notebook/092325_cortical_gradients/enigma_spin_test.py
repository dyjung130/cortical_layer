#!/usr/bin/env python
"""
Wrapper script to run ENIGMA spin test
This script is intended to be executed within the 'lami' Python environment, which is pre-configured for use with the ENIGMA toolbox.
This wrapper script was created because the gradient analysis code and ENIGMA toolbox use different python version 3.9 vs 3.11
"""
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from enigmatoolbox.permutation_testing import spin_test

try:
    # Read input arguments
    map1_file = sys.argv[1]
    map2_file = sys.argv[2]
    n_rot = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    
    # Load data
    map1 = np.load(map1_file)
    map2 = np.load(map2_file)
    
    # Run spin test using the ENIGMA toolbox
    p_spin, r_dist = spin_test(map1, map2, 
                               surface_name='fsa5',
                               parcellation_name='schaefer_400',
                               n_rot=n_rot,
                               null_dist=True)
    
    # Output only JSON to stdout
    result = {
        'p_value': float(p_spin),
        'null_dist': r_dist.tolist(),
        'success': True
    }
    
    print(json.dumps(result)) 
    
except Exception as e:
    error_result = {
        'success': False,
        'error': str(e),
        'error_type': type(e).__name__
    }
    print(json.dumps(error_result))
    sys.exit(1)