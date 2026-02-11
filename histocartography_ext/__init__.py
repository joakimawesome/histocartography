import sys
from .pipeline import PipelineRunner, BatchPipelineRunner

# Backward compatibility hack for unpickling legacy checkpoints
# that expect 'histocartography' module
sys.modules['histocartography'] = sys.modules[__name__]


__all__ = [
    'PipelineRunner',
    'BatchPipelineRunner'
]
