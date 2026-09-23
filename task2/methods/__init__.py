from task2.methods.cdan import CDAN
from task2.methods.dan import DAN
from task2.methods.dann import DANN
from task2.methods.source_only import SourceOnly

METHODS = {m.name: m for m in (SourceOnly, DAN, DANN, CDAN)}
