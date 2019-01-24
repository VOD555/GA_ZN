import random
import shutil
import os
import sys
import time
import pandas as pd
import numpy as np
import gromacs as gw
import fep
import math

def Initial(n):
    # genrate initial population
    dir_home = os.getcwd()
    results = pd.DataFrame()
    names = []
    for i in range(n):
        name = 'set'+str(i)
        shutil.copytree('template', name)
        sigma = 0.04 + (0.4-0.04)*random.random()
        epsilon = 10.0 + 30*random.random()
        with gw.utilities.in_dir(name, create=False):
            DeltaG, IOD, Score = score(sigma, epsilon)
        result = pd.DataFrame({'sigma': sigma, 'epsilon': epsilon,
                               'DeltaG': DeltaG, 'IOD': IOD, 'Score': Score},
                               index=[0])
        results = results.append(result)
        results.to_pickle('tt1.pkl')
        names.append(name)

    ranked, best4 = rank(results)
    ranked.to_pickle('gen0.pkl')
    for name in names:
        shutil.rmtree(name)
    return best4

def rank(results):
    ranked = results.sort_values(by = ['Score'], ascending= False)
    best4 = ranked.iloc[0:4]
    return ranked, best4

def breed(n, parents):
    # n number of offsprings
    # parents: DataFrame of parent parameters
    offsprings = pd.DataFrame()
    kids = 0
    while kids < n:
        n_ids = len(parents.index)
        ids = range(n_ids)
        pa_id = ids.pop(random.randrange(0,n_ids))
        ma_id = ids.pop(random.randrange(0,n_ids-1))
        pa = parents.iloc[pa_id]
        ma = parents.iloc[ma_id]
        p_epsilon = random.random()
        p_sigma = random.random()

        if p_epsilon < 0.5:
            r = random.random()
            new_epsilon = r*pa['epsilon']+(1-r)*ma['epsilon']
        if p_sigma < 0.5:
            r = random.random()
            new_sigma = r*pa['sigma']+(1-r)*ma['sigma']
        if (p_epsilon < 0.5) and (p_sigma < 0.5):
            kids += 1
            offspring = pd.DataFrame({'sigma': new_sigma,
                                      'epsilon': new_epsilon},
                                      index = [0] )
            offsprings = offsprings.append(offspring)
    return offsprings

def mutation(origin):
    # parameter_set: a DataFrame
    mutated = pd.DataFrame()
    for i in origin.index:
        p = random.random()
        if p <0.25:
            pp = random.random()
            if pp < 0.5:
                sigma = 0.04 + (0.4-0.04)*random.random()
                epsilon = origin.iloc[i]['epsilon']
            else:
                epsilon = 10.0 + 30*random.random()
                sigma = origin.iloc[i]['sigma']
            mutate = pd.DataFrame({'sigma': sigma, 'epsilon': epsilon},
                                   index=[0])
            mutated = mutated.append(mutate)
    return mutated

def new_generation(best4):
    offsprings = breed(4, best4)
    new_gen = best4.append(offsprings)
    mutated = mutation(new_gen)
    new_gen = new_gen.append(mutated)
    names = []
    new_gen.index = range(0, len(new_gen.index))
    for i in new_gen.index:
        if math.isnan(new_gen['Score'][i]):
            sigma = new_gen['sigma'][i]
            epsilon = new_gen['epsilon'][i]
            name = 'set'+str(i)
            shutil.copytree('template', name)
            with gw.utilities.in_dir(name, create=False):
                DeltaG, IOD, Score = score(sigma, epsilon)
            new_gen.at[str(i), 'DeltaG'] = DeltaG
            new_gen.at[str(i), 'IOD'] = IOD
            new_gen.at[str(i), 'Score'] = Score
            names.append(name)
    for name in names:
        shutil.rmtree(name)
    return new_gen

def score(sigma, epsilon):
    # change ZND parameters
    ffnonpath = 'settings/charmm36-jul2017.ff/ffnonbonded.itp'
    with open(ffnonpath, 'r') as ff:
        data = ff.readlines()
    data[480] = ' ZND    0   47.390000  -1.00    A   {0:.7f}  {1:.13f}\n'.format(sigma, epsilon)
    with open(ffnonpath, 'w') as ff:
        ff.writelines( data )

    gwat, IOD = SFE_RDF()
    DeltaG = gwat.results.DeltaA.Gibbs
    G = DeltaG.value
    Score = np.exp(-np.abs(G+2022.127)/50.0)*0.7 + np.exp(-np.abs(2.08-IOD)*10.0)*0.3
    return DeltaG, IOD, Score

def bar_state(gwat):
    complete = True
    for component, lambdas in gwat.lambdas.items():
        for l in lambdas:
            dir = gwat.wdir(component, l)+'/md.part0001.xtc'
            complete = complete and os.path.exists(dir)
    return complete

def SFE_RDF():
    from subprocess import call
    from mdpow.run import equilibrium_simulation
    from mdpow.config import get_configuration

    #working directory
    dir_home = os.getcwd()

    #from config file get setting options
    runfile = 'runinput.yml'
    cfg = get_configuration(runfile)

    distance = cfg.get('setup', 'distance')

    boxtype = cfg.get('setup', 'boxtype')

    solvent = "water"
    solventmodel = cfg.get('setup', 'watermodel')

    import mdpow.equil

    topdir = cfg.get("setup", "name")
    #set directory
    dirname = os.path.join(topdir, mdpow.equil.WaterSimulation.dirname_default)
    #setup system and energy minimization
    S = mdpow.equil.WaterSimulation(molecule="ZNM", distance=distance, solvent=solvent, solventmodel=solventmodel, dirname=dirname)
    S.topology(itp='ZNM.itp', top_template='settings/system_charmm.top')
    S.solvate(struct="znm.pdb")
    S.energy_minimize(mdp='settings/em_opls.mdp')

    EM_dirname = os.path.join(topdir, S.dirname_default)

    S.MD_relaxed(runtime=10, qscript=['NPT.sge'])
    relaxed_dir = EM_dirname+'/MD_relaxed/'
    with gw.utilities.in_dir(relaxed_dir, create=False):
        cmd = ['qsub', 'NPT.sge']
        rc = call(cmd)

    relaxed_gro = relaxed_dir+'md.gro'
    while not os.path.exists(relaxed_gro):
        time.sleep(60)

    S.MD(runtime=50, mdp='settings/NPT_opls.mdp', qscript=['NPT.sge'])
    NPT_dir = EM_dirname+'/MD_NPT/'
    with gw.utilities.in_dir(NPT_dir, create=False):
        cmd = ['qsub', 'NPT.sge']
        rc = call(cmd)

    NPT_gro = NPT_dir+'md.gro'
    #wait job completed
    while not os.path.exists(NPT_gro):
        time.sleep(60)

    fep_dirname = os.path.join(topdir, fep.Ghyd.dirname_default)

    schedules = {'coulomb': fep.FEPschedule.load(cfg, "FEP_schedule_Coulomb"),
                 'vdw': fep.FEPschedule.load(cfg, "FEP_schedule_VDW"),
                }

    gwat = fep.Ghyd(simulation=S, runtime=20, mdp='settings/bar_opls.mdp', schedules=schedules, qscript=['GAMDPOW.sge'], dirname=fep_dirname)
    gwat.setup(qscript=['GAMDPOW_SETUP.sge'])
    with gw.utilities.in_dir(gwat.dirname, create=False):
        for component, scripts in gwat.scripts.items():
            s = scripts[1] # relative to dirname
            cmd = ['qsub', s]
            rc = call(cmd)
    #check whether simulation completed

        complete = bar_state(gwat)
        while not complete:
            time.sleep(60)
            complete = bar_state(gwat)

    gwat.analyze()

    # RDF calculation
    import MDAnalysis as mda
    top = fep_dirname+'/Coulomb/0000/md.part0001.gro'
    trj = fep_dirname+'/Coulomb/0000/md.part0001.xtc'
    u = mda.Universe(top, trj)

    from MDAnalysis.analysis.rdf import InterRDF
    g1 = u.select_atoms('name ZND')
    g2 = u.select_atoms('name OW')
    RDF = InterRDF(g1,g2, bins = 100, range=(0.0, 5.0))
    RDF.run()
    df = pd.DataFrame({'bins': RDF.bins, 'RDF': RDF.rdf})
    df2 = df.sort_values(by = ['RDF'])
    IOD = df2['bins'].iat[-1]

    return gwat, IOD

gen1st = Initial(8)
