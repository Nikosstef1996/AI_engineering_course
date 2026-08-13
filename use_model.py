
import pickle 

import numpy as np

local_classifier = pickle.load(open('classifier.pickle','rb'))
local_scaler = pickle.load(open('sc.pickle','rb'))

new_pred = local_classifier.predict(local_scaler.transform(np.array([[40,20000]])))

