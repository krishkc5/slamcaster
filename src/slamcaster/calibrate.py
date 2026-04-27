 from __future__ import annotations
 
 from typing import Any
 
 import numpy as np
 from sklearn.calibration import CalibratedClassifierCV
 
 
 def calibrate_model(
     model: Any,
     X_val: Any,
     y_val: np.ndarray,
     method: str = "isotonic",
 ) -> Any:
     """Calibrate a probabilistic classifier using validation data."""
     cal = CalibratedClassifierCV(model, method=method, cv="prefit")
     cal.fit(X_val, y_val)
     return cal
