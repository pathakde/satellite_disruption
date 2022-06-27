#!/usr/bin/env python

import pandas as pd
import tangos
import tangos.examples.mergers as mergers
import pynbody

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
h = 0.6776942783267969

from tangos_halo_module.halo_properties import track_halo_property, get_timesteps, ID_to_sim_halo_snap, infall_final_n_particles, infall_final_coordinates, apocentric_distance, disruption_time, accretion_time, orbit_interpolation, infall_velocity, quenching_time, max_sSFR_time, max_mass_time 
from tangos_halo_module.path import get_file_path, get_halo_snap_num, read_file
from tangos_halo_module.halos import ID_to_tangos_halo, get_survivors, get_main_progenitor_branch, get_zombies, get_host, get_survivor_IDs, get_zombie_IDs, blockPrint, enablePrint, tangos_to_pynbody_halo

data = pd.read_csv('Data100.csv')
ids = np.asarray(data['ID'])
sim = np.asarray(data['Simulation'])

for sims in ['h329', 'h242', 'h229', 'h148']: #['h329', 'h148']:
    idx = ids[sim==sims]
    column_names = ['snapshot']+list(idx)
    df = pd.DataFrame(columns = column_names)
    df.to_csv(str(sims)+'MPB_ids.csv', index=False)
    ts = get_timesteps(simulation=sims, resolution=100)[3]
    df['snapshot'] = ts
    df.to_csv(str(sims)+'MPB_ids.csv', index=False)
    print('initialized ', sims)

resolution=100
def store_MPB_ids(ID=0):
    if ID != 0:
        print('Started ', ID)
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
        print(simulation, status, halo_id, snap_num)
        tangos_halo = ID_to_tangos_halo(ID=ID, resolution=resolution)
    
    # Satellite MPB ids
    MPB, MPB_ids = get_main_progenitor_branch(tangos_halo=tangos_halo, simulation=simulation, resolution=resolution)
    print(MPB_ids)
    print(MPB)
    # Store and recall
    # Host MPB and ids
    host = get_host(simulation=simulation, resolution=resolution)
    host_MPB, host_ids = get_main_progenitor_branch(tangos_halo=host, simulation=simulation, 
                                                    resolution=resolution)
    print(host_ids)
    # Only consider halo while separate from host
    for i, idx in enumerate(MPB_ids):
        if idx in host_ids:
            MPB_ids[i]=0
            
    MPB_ids[MPB_ids==host_ids] = 0
    print('Separate from host', MPB_ids)

    for i in range(len(MPB_ids)):
        if MPB_ids[i]!=0:
            if len(str(MPB[i]))>6:
                MPB_ids[i]=get_halo_snap_num(tangos_halo=MPB[i])[2]
            else: 
                MPB_ids[i]=0
#     dat = {str(ID): MPB_ids,}
#     df = pd.DataFrame(data=dat)
    df = pd.read_csv(str(simulation)+'MPB_ids.csv')
    df[str(ID)] = MPB_ids
    df.to_csv(str(simulation)+'MPB_ids.csv', index=False)
    print(df)
    print('Finished ', ID)
    
completed_ids = np.zeros(0) 

# Run code to get MPB ids in a loop
count=0
done=0
while done<=len(ids):
    try:
        print('Watch me go for the ', count, 'th time...')
        for idx in ids:
            if idx in completed_ids:
                print('Already Exists ', idx)
                done=len(completed_ids)
                print('Completed Count --> ', done)
                pass
            else:
                print('Started ', idx)
                store_MPB_ids(ID=idx)
                completed_ids = np.concatenate((completed_ids, [idx]), axis=None)
                print('Finished ', idx)
                done=len(completed_ids)
                print('Completed Count --> ', done)
    except: 
        count+=1
        continue