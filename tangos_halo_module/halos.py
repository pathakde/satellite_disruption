import numpy as np
import h5py
import re

import tangos
import tangos.examples.mergers as mergers

from .path import get_file_path, get_halo_snap_num
from .halo_properties import get_main_progenitor_branch, get_timesteps

import sys, os
# Disable printing
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore printing
def enablePrint():
    sys.stdout = sys.__stdout__

    
    
def tangos_to_pynbody_halo(tangos_halo, simulation, resolution=1000):
    '''
    input params: 
        tangos_halo: a valid Tangos halo object
        simulation: h148, h229, h242, h329: string
    output params:
        the corresponding valid Pynbody halo object
    '''
    #edit local path to database folder
    if resolution==1000:
        tangos.init_db("simulation_database/"+ str(simulation) +".db")
    elif resolution==100:
        tangos.init_db("/data/Sims/Databases/100Nptcls/"+ str(simulation) +".db")
    else:
        raise ValueError("Resolution not implemented yet.")
    
    if simulation == 'h148':
        #edit path to local simulation path
        tangos.config.base = '/home/pathakde/MAP2021/Sims/'+ str(simulation) +'.cosmo50PLK.3072g/'+ str(simulation) +'.cosmo50PLK.3072g3HbwK1BH/'
    else:
        tangos.config.base = '/home/pathakde/MAP2021/Sims/'+ str(simulation) +'.cosmo50PLK.3072g/'+ str(simulation) +'.cosmo50PLK.3072gst5HbwK1BH/'
    return tangos_halo.load()



def ID_to_tangos_halo(ID=0, simulation=0, status=0, halo_id=0, snap_num=0, resolution=1000):
    '''
    input params: 
        ID: new halo ID of the format SSSXXXXYYYY [3# sim] 
                                                 +[1-4# latest snap num when halo exists] 
                                                 +[4# sim halo ID at latest snap]
    output params: 
       tangos halo: corresponding tangos halo object
        
    '''
    import re
    
    if simulation != 0:
        sim = str(simulation)
    elif ID != 0:
        ID = int(ID)
        sim = 'h'+str(ID)[:3] #eg. h329    
    else:
        raise ValueError("Cannot determine simulation. You got this tho.")
    # Now we can get the simulation and timesteps 
    #edit local path to database folder
    if resolution==1000:
        tangos.init_db("simulation_database/"+ str(sim) +".db")
    elif resolution==100:
        tangos.init_db("/data/Sims/Databases/100Nptcls/"+ str(sim) +".db")
    else:
        raise ValueError("Resolution not implemented yet.")
    timesteps = tangos.get_simulation("snapshots").timesteps 
    snap_nums = [re.findall(r'.00+[\d]+', str(snap))[0][3:] for snap in timesteps]
    
    if snap_num != 0:
        snap_num = str(snap_num)
    elif ID != 0:
        snap_num = str(str(ID)[3:7])
    elif status == 'Survivor' or status == 'Host':
        snap_num = str(4096)
    else:
        raise ValueError("Cannot determine snapshot number. You got this tho.")
    # check if strings equal
    for index, t in enumerate(snap_nums):
        if t == snap_num:
            timestep = timesteps[index]

    if halo_id != 0:
        halo_id = halo_id
    elif ID != 0:
        halo_id = str(ID)[7:]
    else:
        raise ValueError("Cannot determine halo number at given snapshot. You got this tho.")

    tangos_halo = timestep[halo_id]
    return tangos_halo




def get_host(simulation, resolution=1000):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
    output params:
        host: the main Milky Way analog in a simulation: a Tangos Halo object
    '''
    #edit .db path to match local address
    #edit local path to database folder
    if resolution==1000:
        tangos.init_db("simulation_database/"+ str(simulation) +".db")
    elif resolution==100:
        tangos.init_db("/data/Sims/Databases/100Nptcls/"+ str(simulation) +".db")
    else:
        raise ValueError("Resolution not implemented yet.")
    # Host
    if simulation == 'h148':
        halo = tangos.get_halo("snapshots/"+ str(simulation) +
                               ".cosmo50PLK.3072g3HbwK1BH.004096/halo_1")
    else:
        halo = tangos.get_halo("snapshots/"+ str(simulation) +
                               ".cosmo50PLK.3072gst5HbwK1BH.004096/halo_1")
    return halo



def get_survivor_IDs(simulation, resolution=1000):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
    output params:
        survivor IDs: IDs of unique satellites (with at least 1 star) that survive at z=0: a list of custom halo IDs
    '''
    host = get_host(simulation, resolution=resolution)
    if resolution==1000:
        tangos.init_db("simulation_database/"+ str(simulation) +".db")
    elif resolution==100:
        tangos.init_db("/data/Sims/Databases/100Nptcls/"+ str(simulation) +".db")
    else:
        raise ValueError("Resolution not implemented yet.")
    # Survivors
    survivors = []
    for sats in host['childHalo']:
        if sats.NStar > 0:
            survivors.append(sats)
    survivors = list(set(survivors))

    IDs = []        
    for sat in survivors:
        IDs.append(get_halo_snap_num(tangos_halo=sat)[2])
    IDs = list(set(IDs))
    host_ID = get_halo_snap_num(tangos_halo=host)[2]
    if host_ID in IDs:
        IDs.remove(host_ID)
    #IDs
    IDs.sort()
    return IDs



def get_survivors(simulation, resolution=1000):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
    output params:
        survivors: satellites (with at least 1 star) that survive z=0: a list of Tangos Halo objects
    '''
    blockPrint()
    
    # Survivors
    survivor_IDs = get_survivor_IDs(simulation=simulation, resolution=resolution)
    print(survivor_IDs)
    if resolution==1000:
        tangos.init_db("simulation_database/"+ str(simulation) +".db")
    elif resolution==100:
        tangos.init_db("/data/Sims/Databases/100Nptcls/"+ str(simulation) +".db")
    else:
        raise ValueError("Resolution not implemented yet.")
    survivors = []
    for idx in survivor_IDs:
        halo = ID_to_tangos_halo(ID=idx, simulation=0, status=0, halo_id=0, snap_num=0, resolution=resolution)
        survivors.append(halo)
        print(halo)
    enablePrint()
    return survivors


def get_zombie_IDs(simulation, resolution=1000):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
    output params:
        zombie IDs: IDs of unique satellites (with at least 1 star) that disrupt before z=0: a list of custom halo IDs
    '''
    # For some very strange reason, the function keeps breaking at different spots
    # Unless these print statements exist where they currently do
    # So... This happened.
    blockPrint()
    print("This won't print")
    # And this will enable printing again
    # If you ever need it
    #enablePrint()
    
    # Now this is the actual code.
    # Note: Getting MPB corrupts halo object (god knows how)

    if resolution==1000:
        tangos.init_db("simulation_database/"+ str(simulation) +".db")
    elif resolution==100:
        tangos.init_db("/data/Sims/Databases/100Nptcls/"+ str(simulation) +".db")
    else:
        raise ValueError("Resolution not implemented yet.")
    t_Gyr = get_timesteps(simulation=simulation, resolution=resolution)[0]
    print(t_Gyr)
    # Host
    host = get_host(simulation=simulation, resolution=resolution)
    print(host)

    # Zombies
    redshift, ratio, progenitor_halos = mergers.get_mergers_of_major_progenitor(host)
    zombies = [x[1] for x in progenitor_halos] 
    print(zombies)
    host_MPB, host_MPB_ids = get_main_progenitor_branch(tangos_halo=host, simulation=simulation, resolution=resolution)
    print(host_MPB_ids)
    latest_signature = []
    for sat in zombies: 
        print(sat['Mvir'])
        print(sat)
        sat_MPB, sat_MPB_ids = get_main_progenitor_branch(tangos_halo=sat, simulation=simulation, 
                                                          resolution=resolution)
        print(sat_MPB_ids)
        print(sat_MPB)
        mismatch_times = t_Gyr[(sat_MPB_ids!=host_MPB_ids)&(sat_MPB_ids!=0)]
        if len(mismatch_times)>0:
            latest_signature_time = max(mismatch_times)
            print(latest_signature_time)
            latest_sig = sat_MPB[(t_Gyr==latest_signature_time)]
            
            if len(str(latest_sig))>10: # check if it is not None
                print(latest_sig)
                latest_signature.append(latest_sig[0])

    IDs = []        
    for sat in latest_signature:
        IDs.append(get_halo_snap_num(tangos_halo=sat)[2])
    IDs = list(set(IDs))
    host_ID = get_halo_snap_num(tangos_halo=host)[2]
    if host_ID in IDs:
        IDs.remove(host_ID)
    #IDs
    IDs.sort()
    enablePrint()
    return IDs



def get_zombies(simulation, resolution=1000):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
    output params:
        survivors: satellites (with at least 1 star) that disrupt before z=0: a list of Tangos Halo objects
    '''
    blockPrint()
    # Zombies
    zombie_IDs = get_zombie_IDs(simulation = simulation, resolution=resolution)
    zombies = []
    for idx in zombie_IDs:
        halo = ID_to_tangos_halo(ID=idx, simulation=0, status=0, halo_id=0, snap_num=0, resolution=resolution)
        zombies.append(halo)
        print(halo)
    enablePrint()
    return zombies