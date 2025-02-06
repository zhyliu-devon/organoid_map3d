import sys
import warnings

# warnings.warn(
#     "organoidContour is being renamed to organoid_mapping. "
#     "Please update your imports.",
#     FutureWarning
# )

# # Alias the package name so both old and new names work
sys.modules["organoidContour"] = sys.modules["organoid_contour"] 
