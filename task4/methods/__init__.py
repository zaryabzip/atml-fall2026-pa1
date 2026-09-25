from task4.methods.gcsc import GCSC
from task4.methods.proser import PROSER
from task4.methods.rpl import RPL
from task4.methods.vanilla import Vanilla

METHODS = {m.name: m for m in (Vanilla, GCSC, PROSER, RPL)}
