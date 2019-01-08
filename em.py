import mdpow.equil
S = mdpow.equil.WaterSimulation(molecule="ZNM", distance=1.2)
S.topology(itp='ZNM.itp', top_template='settings/system_charmm.top')
S.solvate(struct="znm.pdb")
S.energy_minimize()
S.MD_relaxed(runtime=5)

import gromacs
r = gromacs.run.MDrunner(dirname=S.dirs['MD_relaxed'], deffnm="md", c="md.pdb", cpi=True, append=True, v=True)
r.run()

S.MD(runtime=10, qscript=['local.sh'])
r = gromacs.run.MDrunner(dirname=S.dirs['MD_NPT'], deffnm="md", c="md.pdb", cpi=True, append=True, v=True)
r.run()

import mdpow.fep
gwat = mdpow.fep.Ghyd(simulation=S, runtime=10)
gwat.setup()
