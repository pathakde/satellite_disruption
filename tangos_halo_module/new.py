import h5py
import pandas as pd
import numpy as np
from .path import get_file_path, ID_to_sim_halo_snap


def unique_tracks(ID=0, key='M_star_|_Msol', simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, resolution=100):
    '''
    input params:
        ID: full unique halo ID
        key: Halo File Key
        resolution: 100 [near-mint], 1000 [near-mint], Mint [mint]
    output params:
        maximum stellar mass: the maximum stellar mass of a dwarf before disruption
            [units: Msol]
        time of maximum stellar mass: the earliest time before disruption when a dwarf reaches its maximum stellar mass
            [units: Gyr since start of simulation]
    '''
    if ID != 0:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
    #Host
    path = get_file_path(tangos_halo=0, simulation=simulation, status='Host', halo_id='1', snap_num='4096', resolution=resolution)
    with h5py.File(path, 'r') as fh:
        hostMvir = fh['Mvir_|_Msol'][:]
    #Satellite
    path = get_file_path(tangos_halo=tangos_halo, simulation=simulation, status=status, halo_id=halo_id, snap_num=snap_num, resolution=resolution)
    with h5py.File(path, 'r') as f:
        if key in f.keys():
            t_Gyr = f['time_|_Gyr'][:]
            satMvir = f['Mvir_|_Msol'][:]
            sat_track = f[key][:]
            return t_Gyr[(satMvir>0) & (satMvir!=hostMvir)], sat_track[(satMvir>0) & (satMvir!=hostMvir)]
        else:
            raise ValueError('Key ', str(key), ' does not exist in Halo file ', str(path)) 
            
def get_mass_profiles(ID=0, ptcl='D', simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, resolution=100):
    '''
    input params:
        ID: full unique halo ID
        ptcl: D->dark matter, S->star
        resolution: 100 [near-mint], 1000 [near-mint], Mint [mint]
    output params:
        maximum stellar mass: the maximum stellar mass of a dwarf before disruption
            [units: Msol]
        time of maximum stellar mass: the earliest time before disruption when a dwarf reaches its maximum stellar mass
            [units: Gyr since start of simulation]
    '''
    if ID != 0:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
    #Satellite profile data
    if resolution==100:
        file = "/home/pathakde/MAP2021/mergers/Halo_Files/Halo_Profiles/"+str(simulation)+"MPB_R"+str(ptcl)+"mass.csv"
    elif resolution=='Mint':
        file = "/home/pathakde/MAP2021/mergers/Mint_Data/Halo_Files/Halo_Profiles/"+str(simulation)+"MPB_R"+str(ptcl)+"mass.csv"
    df = pd.read_csv(file)
    data = df[str(ID)].to_numpy()
    data = np.array([eval(x) for x in data])
    
    #Satellite time array
    path = get_file_path(tangos_halo=tangos_halo, simulation=simulation, 
                         status=status, halo_id=halo_id, snap_num=snap_num, resolution=resolution)
    with h5py.File(path, 'r') as f:
        t_Gyr = f['time_|_Gyr'][:]
    return t_Gyr, data

def get_sSFR_tracks(ID=0, simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, resolution='Mint'):
    '''
    input params:
        ID: full unique halo ID
        ptcl: D->dark matter, S->star
        resolution: 100 [near-mint], 1000 [near-mint], Mint [mint]
    output params:
        maximum stellar mass: the maximum stellar mass of a dwarf before disruption
            [units: Msol]
        time of maximum stellar mass: the earliest time before disruption when a dwarf reaches its maximum stellar mass
            [units: Gyr since start of simulation]
    '''
    if ID != 0:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
    #Satellite profile data
    if resolution==100:
        file = "/home/pathakde/MAP2021/mergers/Halo_Files/Halo_Profiles/"+str(simulation)+"MPB_sSFR.csv"
    elif resolution=='Mint':
        file = "/home/pathakde/MAP2021/mergers/Mint_Data/Halo_Files/sSFR_Tracks/"+str(simulation)+"MPB_sSFR.csv"
    df = pd.read_csv(file)
    data = df[str(ID)].to_numpy()
#     data = np.array([eval(x) for x in data])
    
    #Satellite time array
    path = get_file_path(tangos_halo=tangos_halo, simulation=simulation, 
                         status=status, halo_id=halo_id, snap_num=snap_num, resolution=resolution)
    with h5py.File(path, 'r') as f:
        t_Gyr = f['time_|_Gyr'][:]
    return t_Gyr, data