#!/usr/bin/env conda-env-py3

import pandas as pd
import tangos
import tangos.examples.mergers as mergers

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
h = 0.6776942783267969

from tangos_halo_module.halo_properties import get_main_progenitor_branch, track_halo_property, get_timesteps, ID_to_sim_halo_snap, infall_final_n_particles, infall_final_coordinates, apocentric_distance, disruption_time, accretion_time, orbit_interpolation, infall_velocity, quenching_time, max_sSFR_time, max_mass_time 
from tangos_halo_module.path import get_file_path, get_halo_snap_num, read_file
from tangos_halo_module.halos import ID_to_tangos_halo, get_survivors, get_zombies, get_host, get_survivor_IDs, get_zombie_IDs, blockPrint, enablePrint, tangos_to_pynbody_halo

d = pd.read_csv('Data.csv')
ids = d['IDs'].to_numpy()
print(ids)

# Load routine for getting tracks
def sSFR_track(ID=0, simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0):
    
    from tangos_halo_module.halo_properties import get_main_progenitor_branch
    from tangos_halo_module.halos import tangos_to_pynbody_halo

    if ID != 0:
        print('Started ', ID)
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
        print(simulation, status, halo_id, snap_num)
        tangos_halo = ID_to_tangos_halo(ID=ID)
    
    # Satellite MPB ids
    MPB, MPB_ids = get_main_progenitor_branch(tangos_halo=tangos_halo, simulation=simulation)
    print(MPB_ids)
    print(MPB)
    # Host MPB and ids
    host = get_host(simulation=simulation)
    host_ids = get_main_progenitor_branch(tangos_halo=host, simulation=simulation)[1]

    # Only consider halo while separate from host
    MPB_ids[MPB_ids==host_ids] = 0
    print(MPB_ids)

    for i in range(len(MPB_ids)):
        if MPB_ids[i]!=0:

            MPB_ids[i]=get_halo_snap_num(tangos_halo=MPB[i], simulation=simulation)[2]
    print('MPB new IDs: ', MPB_ids)
    SFR_track = np.zeros(0)
    sSFR_track = np.zeros(0)
    mass_track = np.zeros(0)
    new_mass_track = np.zeros(0)

    for idx in MPB_ids:
        print(idx)
        if idx==0:
            SFR_track = np.concatenate((SFR_track, [-1]), axis=None)
            sSFR_track = np.concatenate((sSFR_track, [-1]), axis=None)
            mass_track = np.concatenate((mass_track, [-1]), axis=None)
            new_mass_track = np.concatenate((new_mass_track, [-1]), axis=None)
        else:
            tang_halo = ID_to_tangos_halo(ID=idx)
            print(tang_halo)
            pyn_halo = tangos_to_pynbody_halo(tangos_halo=tang_halo, simulation=simulation)
            mass = np.asarray(pyn_halo.s['mass'])
            age = pyn_halo.properties['time'].in_units('Gyr') - pyn_halo.s['tform'].in_units('Gyr') #SimArray in Gyr
            new_mass = np.sum(mass[age <= 0.1]) # Formed within last 100 Myr
            total_mass = np.sum(mass)
            SFR = new_mass/100
            sSFR = SFR/total_mass
            SFR_track = np.concatenate((SFR_track, [SFR]), axis=None)
            sSFR_track = np.concatenate((sSFR_track, [sSFR]), axis=None)
            mass_track = np.concatenate((mass_track, [total_mass]), axis=None)
            new_mass_track = np.concatenate((new_mass_track, [new_mass]), axis=None)
    print(SFR_track, sSFR_track, mass_track, new_mass_track)
    dat = {'IDs': ID, 
           'SFR': [SFR_track], 
           'sSFR': [sSFR_track], 
           'stellar_mass': [mass_track], 
           'new_stellar_mass': [new_mass_track],}
    df = pd.DataFrame(data=dat)
    return df
    
Initialize dataframe
column_names = ['IDs', 'SFR', 'sSFR', 'stellar_mass', 'new_stellar_mass']
df = pd.DataFrame(columns = column_names)
df.to_csv('sSFR_data.csv', index=False)
# df

df = pd.read_csv('sSFR_data.csv')
completed_ids = df['IDs'].to_numpy()


for idx in ids:
    if idx in completed_ids:
        print('Already Exists ', idx)
        df.to_csv('sSFR_data.csv', index=False)
        pass
    else:
        sat_df = sSFR_track(ID=idx)
        #df = pd.concat([df, sat_df])
        sat_df.to_csv('sSFR_data.csv', mode='a', index=False)
        print('Finished ', idx)
        
