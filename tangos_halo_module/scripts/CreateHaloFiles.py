#!/usr/bin/env conda-env-py3

import tangos
import os

import numpy as np
import h5py
h = 0.6776942783267969

from tangos_halo_module.path import get_file_path, get_halo_snap_num
from tangos_halo_module.halos import get_survivors, get_zombies, get_host, get_survivor_IDs, get_zombie_IDs, blockPrint, enablePrint
from tangos_halo_module.halo_properties import track_halo_property, get_timesteps

def track_halo_property(simulation, key, tangos_halo=0, halo_path=0, halo_id=0, snap_num=0, resolution=1000):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
        key: pre-existing tangos halo property (eg. 'Mvir' or 'VXc'): string
        tangos_halo: a valid Tangos halo object
        halo_path: 0 or a complete tangos halo address: path or string object
        halo_id: halo id: string or numeric
        snap_num: 4 digit simulation snapshot number: string
    output params:
        tracked parameter: earliest --> latest snapshot: numpy array
    '''
    
    if tangos_halo:
        halo = tangos_halo
    elif halo_id and snap_num and simulation:
        #string tangos halo query from components
        if simulation == 'h148':
            halo = tangos.get_halo("snapshots/"+ str(simulation) +
                               ".cosmo50PLK.3072g3HbwK1BH.00"+ str(snap_num) +"/"+ str(halo_id))
        else:
            halo = tangos.get_halo("snapshots/"+ str(simulation) +
                               ".cosmo50PLK.3072gst5HbwK1BH.00"+ str(snap_num) +"/"+ str(halo_id))
    elif halo_path:
        halo = tangos.get_halo(str(halo))
    else:
        raise ValueError("Halo %r not found. You got this tho." % (halo))

        #edit .db path to match local address
    if resolution==1000:
        tangos.init_db("simulation_database/"+ str(simulation) +".db")
    elif resolution==100:
        tangos.init_db("/data/Sims/Databases/100Nptcls/"+ str(simulation) +".db")
    else:
        raise ValueError("Resolution not implemented yet.")
        
    # round all times in Gyr to 2 decimal places due to 
    # inconsistent float truncation between response to livecalculation query and direct database query
    halo_gyr = round(halo.timestep.time_gyr, 2)
    prog = np.flip(halo.calculate_for_progenitors(str(key))[0])
    desc = halo.calculate_for_descendants(str(key))[0][1:]
    track = np.array(np.concatenate((prog, desc), axis=None))

    len_track = len(track)
    timesteps = get_timesteps(simulation)[0] 
    timesteps = [round(t, 2) for t in timesteps] #in gyr since start of sim
    num_timesteps = len(timesteps)
    #len(tangos.get_simulation("snapshots").timesteps) #62 for h229, h242, h329
    
    if len_track == num_timesteps:
        return np.asarray(track)
    elif len_track > num_timesteps/2: #pad track with initial 0s
        return np.array(np.concatenate((np.zeros(num_timesteps - len_track), track), axis=None), dtype=float)
    else:
        # i = np.where(timesteps==halo_gyr)[0][0]
        i = timesteps.index(halo_gyr)
        new_track = np.zeros(num_timesteps)
        new_track[i+1:i+1+len(desc)] = desc
        new_track[i+1-len(prog):i+1] = prog
        return new_track 
    
def write_halo_file(halo, simulation, status, mode='add', resolution=100):
    import h5py
    h = 0.6776942783267969
    
    if mode == 'write': # Write
        m = 'w'
    else: # Add
        m = 'a'
    path = get_file_path(tangos_halo=halo, simulation=simulation, status=status, halo_id=0, snap_num=0, resolution=resolution)
    with h5py.File(path, m) as f:
        d1 = f.create_dataset('time_|_Gyr', data = get_timesteps(simulation=simulation, resolution=resolution)[0])
        d2 = f.create_dataset('time_|_redshift', data = get_timesteps(simulation=simulation, resolution=resolution)[1])
        d3 = f.create_dataset('time_|_a', data = get_timesteps(simulation=simulation, resolution=resolution)[2])
#         d4 = f.create_dataset('SFR_100Myr_|_Msol/yr', data = track_halo_property(tangos_halo=halo, simulation=simulation, key='SFR_100Myr', resolution=resolution))
        d5 = f.create_dataset('Xc_|_kpc', data = track_halo_property(tangos_halo=halo, simulation=simulation, key='Xc', resolution=resolution)*a/h)
        d6 = f.create_dataset('Yc_|_kpc', data = track_halo_property(tangos_halo=halo, simulation=simulation, key='Yc', resolution=resolution)*a/h)
        d7 = f.create_dataset('Zc_|_kpc', data = track_halo_property(tangos_halo=halo, simulation=simulation, key='Zc', resolution=resolution)*a/h)
        d8 = f.create_dataset('Rvir_|_kpc', data = track_halo_property(tangos_halo=halo, simulation=simulation, key='Rvir', resolution=resolution)*a/h)
        d9 = f.create_dataset('Mvir_|_Msol', data = track_halo_property(tangos_halo=halo, simulation=simulation, key='Mvir', resolution=resolution)*h)
        d10 = f.create_dataset('M_star_|_Msol', data = track_halo_property(tangos_halo=halo, simulation=simulation, key='M_star', resolution=resolution)*h)
        d11 = f.create_dataset('VXc_|_km/s', data = track_halo_property(tangos_halo=halo, simulation=simulation, key='VXc', resolution=resolution))
        d12 = f.create_dataset('VYc_|_km/s', data = track_halo_property(tangos_halo=halo, simulation=simulation, key='VYc', resolution=resolution))
        d13 = f.create_dataset('VZc_|_km/s', data = track_halo_property(tangos_halo=halo, simulation=simulation, key='VZc', resolution=resolution))
        d14 = f.create_dataset('n_gas', data = track_halo_property(tangos_halo=halo, simulation=simulation, key='n_gas', resolution=resolution))
        d15 = f.create_dataset('n_star', data = track_halo_property(tangos_halo=halo, simulation=simulation, key='n_star', resolution=resolution))
        d16 = f.create_dataset('n_dm', data = track_halo_property(tangos_halo=halo, simulation=simulation, key='n_dm', resolution=resolution))
        d17 = f.create_dataset('M_gas_|_Msol', data = track_halo_property(tangos_halo=halo, simulation=simulation, key='M_gas', resolution=resolution)*h)
    return


# Write data

for sim in ['h148', 'h229', 'h242', 'h329']:
    
    # Host
    print('Started Host: ' + str(sim))
    halo = get_host(simulation=sim, resolution=100)
    a = get_timesteps(simulation=sim, resolution=100)[2]
    write_halo_file(halo=halo, simulation=sim, status='Host', resolution=100, mode='write')
    print('Finished with Host: ' + str(sim))
    
    print('Started Survivors: ' + str(sim))
    survivors = get_survivors(simulation=sim, resolution=100)
    for sat in survivors:
        write_halo_file(halo=sat, simulation=sim, status='Survivor', resolution=100, mode='write')
    print('Finished with Survivors: ' + str(sim))
    
    print('Started Zombies: ' + str(sim))
    zombies = get_zombies(simulation=sim, resolution=100)
    for sat in zombies:
        write_halo_file(halo=sat, simulation=sim, status='Zombie', resolution=100, mode='write')
    print('Finished with Zombies: ' + str(sim))