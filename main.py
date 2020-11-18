import sys
from bnz.simulation import BnzSimPy

sim = BnzSimPy()
usr_dir='.'
if len(sys.argv)>1: usr_dir=sys.argv[1]
sim.init(usr_dir)
sim.integrate(a)
