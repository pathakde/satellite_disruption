import tangos
import pynbody
import numpy as np
import h5py

from .path import get_file_path
# from .halos import blockPrint, enablePrint


h = 0.6776942783267969


def ID_to_sim_halo_snap(ID=0):
    '''
    input params: 
        ID: new halo ID of the format SSSXXXXYYYY [3# sim] 
                                                 +[4# latest snap num when halo exists] 
                                                 +[1-4# sim halo ID at latest snap]
    output params: 
       simulation: corresponding simulation as string: 'h148', 'h229', 'h242', 'h329'
       status: status of halo as string [as used in .path.get_file_path]: 'Host', 'Survivor', 'Zombie'
       halo_id: corresponding amiga assigned id of halo as string '1', '283', etc.
       snap_num: corresponding snapshot number in format '0071', '4096', etc.
    '''
    simulation = 'h'+str(ID)[:3] #eg. h329        
    snap_num = str(str(ID)[3:7]) #eg. 4096
    halo_id = str(ID)[7:] #eg. 10
    
    #Status:
    if snap_num == '4096':
        if halo_id == '1':
            status='Host'
        else:
            status='Survivor'
    else:
        status='Zombie'
        
    return simulation, status, halo_id, snap_num


def get_timesteps(simulation, resolution=1000): 
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
    output params:
        snapshot Gyrs: earliest --> latest snapshot: numpy array
        snapshot redshift: earliest --> latest snapshot: numpy array
        snapshot scalefactor: earliest --> latest snapshot: numpy array
        snapshot number: last 4 digit string: earliest --> latest snapshot: list
    '''
    import re
    #edit .db path to match local address
    if resolution==1000:
        tangos.init_db("simulation_database/"+ str(simulation) +".db")
    elif resolution==100:
        tangos.init_db("/data/Sims/Databases/100Nptcls/"+ str(simulation) +".db")
    else:
        raise ValueError("Resolution not implemented yet.")

    timesteps = tangos.get_simulation("snapshots").timesteps  
    len_timesteps = len(timesteps) #len=62
    
    gyr_timesteps = np.zeros(len_timesteps)
    redshift_timesteps = np.zeros(len_timesteps)
    snapnum_timesteps = []
    for i, x in enumerate(timesteps):
        gyr_timesteps[i] = x.time_gyr #early univ --> late univ
        redshift_timesteps[i] = x.redshift #early univ --> late univ
        snapnum_timesteps.append(str(re.findall(r'.00+[\d]+', str(x))[0][3:]))
    #return time in  Gyr,           redshift,             scale factor a,  snapshot numbers
    return gyr_timesteps, redshift_timesteps, 1/(1 + redshift_timesteps), snapnum_timesteps
    

    
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
    
    # edit .db path to match local address
    if resolution==1000:
        tangos.init_db("simulation_database/"+ str(simulation) +".db")
    elif resolution==100:
        tangos.init_db("/data/Sims/Databases/100Nptcls/"+ str(simulation) +".db")
    else:
        raise ValueError("Resolution not implemented yet.")
        
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
        raise ValueError("Halo not found. You got this tho.")
        
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
    
    
    
def first_encounter_time(ID=0, simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, resolution=1000):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
        status: 'Zombie', 'Survivor': string
        tangos_halo: 0 or a valid Tangos halo object
        halo_id: halo id: 0 or string or numeric
        snap_num: 0 or 4 digit simulation snapshot number: string
    output params:
        time of first encounter: the earliest time when a dwarf crosses 1.5 x the virial radius of the host
        [units: Gyr since start of simulation]
    '''
    if ID != 0:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
    #Host
    path = get_file_path(tangos_halo=0, simulation=simulation, status='Host', halo_id='1', snap_num='4096', resolution=resolution)
    with h5py.File(path, 'r') as fh:
        a = fh['time_|_a'][:]
        t_Gyr = fh['time_|_Gyr'][:]
        hostx = fh['Xc_|_kpc'][:]
        hosty = fh['Yc_|_kpc'][:]
        hostz = fh['Zc_|_kpc'][:]
        hostRvir = fh['Rvir_|_kpc'][:]

    #Satellite
    path = get_file_path(tangos_halo=tangos_halo, simulation=simulation, status=status, halo_id=halo_id, snap_num=snap_num)
    with h5py.File(path, 'r') as f:
        satx = f['Xc_|_kpc'][:]
        saty = f['Yc_|_kpc'][:]
        satz = f['Zc_|_kpc'][:]
        satMvir = f['Mvir_|_Msol'][:]

    distance = ((hostx-satx)**2 + (hosty-saty)**2 + (hostz-satz)**2)**(1/2) #pkpc
    
    t_first_encounter = min(t_Gyr[(distance<=1.5*hostRvir)&(satMvir>0)]) #Gyr since beginning of simulation
    return t_first_encounter



def max_mass_time(ID=0, simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, category='stars', resolution=100):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
        status: 'Zombie', 'Survivor': string
        tangos_halo: 0 or a valid Tangos halo object
        halo_id: halo id: 0 or string or numeric
        snap_num: 0 or 4 digit simulation snapshot number: string
        category: 'stars' or 'dm' or 'total': string
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
        t_Gyr = fh['time_|_Gyr'][:]
        hostMstar = fh['M_star_|_Msol'][:]
        hostMvir = fh['Mvir_|_Msol'][:]
        hostMgas = fh['M_gas_|_Msol'][:]
    #Satellite
    path = get_file_path(tangos_halo=tangos_halo, simulation=simulation, status=status, halo_id=halo_id, snap_num=snap_num, resolution=resolution)
    with h5py.File(path, 'r') as f:
        satMstar = f['M_star_|_Msol'][:]
        satMvir = f['Mvir_|_Msol'][:]
        satMgas = f['M_gas_|_Msol'][:]
    total_mass=satMstar+satMvir+satMgas    
    
    if category=='stars':
        max_satM = max(satMstar[satMstar != hostMstar])
        max_satM_time = min(t_Gyr[satMstar==max_satM])
    elif category=='dm':
        max_satM = max(satMvir[satMvir != hostMvir])
        max_satM_time = min(t_Gyr[satMvir==max_satM])
    elif category=='total':
        max_satM = max(total_mass[satMstar != hostMstar])
        max_satM_time = min(t_Gyr[total_mass==max_satM])
    elif category=='ratio':
        ratio = satMstar/total_mass
        ratio[total_mass==0]=0
        max_satM = max(ratio[satMstar != hostMstar])
        max_satM_time = min(t_Gyr[ratio==max_satM])
    else:
        raise ValueError('A very specific bad thing happened. Try category= "stars" or "dm".') 
    
    return max_satM, max_satM_time


"""
def accretion_time(ID=0, simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, category='stars', resolution=1000):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
        status: 'Zombie', 'Survivor': string
        tangos_halo: 0 or a valid Tangos halo object
        halo_id: halo id: 0 or string or numeric
        snap_num: 0 or 4 digit simulation snapshot number: string
        category: 'stars' or 'dm': string
    output params:
    Accretion: First time that the satellite crosses (try) 1x (else 1.5x) the virial radius of the host
               The satellite must have some stars at infall.
        mass at accretion: mass of dwarf at accretion
        0: [units: Msol]
        time of accretion: the latest time when a dwarf crosses from outside to inside 1.5 x the virial radius of the host
        1: [units: Gyr since start of simulation]
        mass ratio of satellite at accretion: mass of dwarf at accretion / mass of host at accretion
        2: [units: None]
        mass ratio of host at accretion: mass of host at accretion / mass of host at z=0
        3: [units: None]
    '''
    import pandas as pd
    if ID != 0:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
        halo_id=int(float(halo_id))
    d = pd.read_csv('sSFR_data100.csv')
    ids = d['ID'].to_numpy()
    IDs = d['IDs'].to_numpy()
    unique_ids = IDs[ids==ID][0] # A string of form [a b c ...]
    unique_ids = np.array(list(map(float, unique_ids[1:-1].split())))
    # Accretion threshold at d*hostRvir
    d=1
    #Host
    path = get_file_path(tangos_halo=0, simulation=simulation, status='Host', halo_id='1', snap_num='4096', resolution=resolution)
    with h5py.File(path, 'r') as fh:
        a = fh['time_|_a'][:]
        t_Gyr = fh['time_|_Gyr'][:]
        hostx = fh['Xc_|_kpc'][:]
        hosty = fh['Yc_|_kpc'][:]
        hostz = fh['Zc_|_kpc'][:]
        hostMvir = fh['Mvir_|_Msol'][:]
        hostRvir = fh['Rvir_|_kpc'][:]
        hostMstar = fh['M_star_|_Msol'][:]
    t_Gyr = np.array([round(t, 3) for t in t_Gyr])
    #Satellite
    path = get_file_path(tangos_halo=tangos_halo, simulation=simulation, status=status, halo_id=halo_id, snap_num=snap_num, resolution=resolution)
    with h5py.File(path, 'r') as f:
        satx = f['Xc_|_kpc'][:]
        saty = f['Yc_|_kpc'][:]
        satz = f['Zc_|_kpc'][:]
        satnstar = f['n_star'][:]
        satMvir = f['Mvir_|_Msol'][:]
        satMstar = f['M_star_|_Msol'][:]
        
    distance = ((hostx-satx)**2 + (hosty-saty)**2 + (hostz-satz)**2)**(1/2) #pkpc
    distance_hostRvir = distance/hostRvir
    
    dips = np.zeros(len(distance))
    t_disruption = disruption_time(ID=ID, simulation=simulation, status=status, 
                                   tangos_halo=tangos_halo, halo_id=halo_id, snap_num=snap_num, resolution=resolution)
    if t_disruption == 0: #this is a weird zombie, which I want to exclude
        t_accretion = -1
        M_accretion = [-1]
        Msat_ratio_accretion = [-1]
        Mhost_ratio_accretion = [-1]   
    else:
        for i, val in enumerate(distance):
            if i > 0 and distance[i]<=d*hostRvir[i] and distance[i-1]>d*hostRvir[i-1] and hostMvir[i]!=satMvir[i] and satMstar[i]>0:
                dips[i]=1
            elif i > 0 and distance[i]<=1.5*d*hostRvir[i] and distance[i-1]>1.5*d*hostRvir[i-1] and hostMvir[i]!=satMvir[i] and satMstar[i]>0:
                dips[i]=2
        if 1 in dips:
            t_accretion = min(t_Gyr[dips==1]) #Gyr since beginning of simulation
        else:
            if t_disruption == -1:
                t_accretion = min(t_Gyr[(distance_hostRvir<=d+0.1) & (hostMstar>0) & (satMstar>0) & (hostMstar!=satMstar)])
            else:
                if 2 in dips:
                    t_accretion = min(t_Gyr[dips==2]) #Gyr since beginning of simulation
                else:
                    t_accretion = disruption_time(ID=ID, simulation=simulation, status=status, 
                                                  tangos_halo=tangos_halo, halo_id=halo_id, snap_num=snap_num, resolution=resolution)     
        t_accretion = round(t_accretion, 3)
        id_accretion = np.array(unique_ids)[t_Gyr==t_accretion]
        if category=='stars':
            M_accretion = satMstar[t_Gyr==t_accretion]
            Msat_ratio_accretion = M_accretion/hostMstar[t_Gyr==t_accretion]
            Mhost_ratio_accretion = hostMstar[t_Gyr==t_accretion]/hostMstar[-1]
            # nstar_accretion = satnstar[t_Gyr==t_accretion]
            
        elif category=='dm':
            M_accretion = satMvir[t_Gyr==t_accretion]
            Msat_ratio_accretion = M_accretion/hostMvir[t_Gyr==t_accretion]
            Mhost_ratio_accretion = hostMvir[t_Gyr==t_accretion]/hostMvir[-1]
        else:
            raise ValueError('A very specific bad thing happened. Try category= "stars" or "dm".')
           #0           #1           #2                    #3
    return t_accretion, M_accretion[0], Msat_ratio_accretion[0], int(id_accretion[0]), Mhost_ratio_accretion[0]
"""

def accretion_time(ID=0, simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, category='stars', resolution=1000):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
        status: 'Zombie', 'Survivor': string
        tangos_halo: 0 or a valid Tangos halo object
        halo_id: halo id: 0 or string or numeric
        snap_num: 0 or 4 digit simulation snapshot number: string
        category: 'stars' or 'dm': string
    output params:
    Accretion: First time that the satellite crosses (try) 1x (else 1.5x) the virial radius of the host
               The satellite must have some stars at infall.
        mass at accretion: mass of dwarf at accretion
        0: [units: Msol]
        time of accretion: the latest time when a dwarf crosses from outside to inside 1.5 x the virial radius of the host
        1: [units: Gyr since start of simulation]
        mass ratio of satellite at accretion: mass of dwarf at accretion / mass of host at accretion
        2: [units: None]
        mass ratio of host at accretion: mass of host at accretion / mass of host at z=0
        3: [units: None]
    '''
    import pandas as pd
    if ID != 0:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
        halo_id=int(float(halo_id))
#    d = pd.read_csv('sSFR_data100.csv')
#    ids = d['ID'].to_numpy()
#    IDs = d['IDs'].to_numpy()
#    unique_ids = IDs[ids==ID][0] # A string of form [a b c ...]
#    unique_ids = np.array(list(map(float, unique_ids[1:-1].split())))
    # Accretion threshold at d*hostRvir
    d=1
    #Host
    path = get_file_path(tangos_halo=0, simulation=simulation, status='Host', halo_id='1', snap_num='4096', resolution=resolution)
    with h5py.File(path, 'r') as fh:
        a = fh['time_|_a'][:]
        t_Gyr = fh['time_|_Gyr'][:]
        hostx = fh['Xc_|_kpc'][:]
        hosty = fh['Yc_|_kpc'][:]
        hostz = fh['Zc_|_kpc'][:]
        hostMvir = fh['Mvir_|_Msol'][:]
        hostRvir = fh['Rvir_|_kpc'][:]
        hostMstar = fh['M_star_|_Msol'][:]
    t_Gyr = np.array([round(t, 3) for t in t_Gyr])
    #Satellite
    path = get_file_path(tangos_halo=tangos_halo, simulation=simulation, status=status, halo_id=halo_id, snap_num=snap_num, resolution=resolution)
    with h5py.File(path, 'r') as f:
        satx = f['Xc_|_kpc'][:]
        saty = f['Yc_|_kpc'][:]
        satz = f['Zc_|_kpc'][:]
        satnstar = f['n_star'][:]
        satMvir = f['Mvir_|_Msol'][:]
        satMstar = f['M_star_|_Msol'][:]
        
    distance = ((hostx-satx)**2 + (hosty-saty)**2 + (hostz-satz)**2)**(1/2) #pkpc
    distance_hostRvir = distance/hostRvir
    
    dips = np.zeros(len(distance))
    t_disruption = disruption_time(ID=ID, simulation=simulation, status=status, 
                                   tangos_halo=tangos_halo, halo_id=halo_id, snap_num=snap_num, resolution=resolution)
    if t_disruption == 0: #this is a weird zombie, which I want to exclude
        t_accretion = -1
        M_accretion = [-1]
        Msat_ratio_accretion = [-1]
        Mhost_ratio_accretion = [-1]   
    else:
        for i, val in enumerate(distance):
            if i > 0 and distance[i]<=d*hostRvir[i] and distance[i-1]>d*hostRvir[i-1] and hostMvir[i]!=satMvir[i] and satMstar[i]>0:
                dips[i]=1
            elif i > 0 and distance[i]<=1.5*d*hostRvir[i] and distance[i-1]>1.5*d*hostRvir[i-1] and hostMvir[i]!=satMvir[i] and satMstar[i]>0:
                dips[i]=2
        if 1 in dips:
            t_accretion = min(t_Gyr[dips==1]) #Gyr since beginning of simulation
        else:
            if t_disruption == -1:
                t_accretion = min(t_Gyr[(distance_hostRvir<=d+0.1) & (hostMstar>0) & (satMstar>0) & (hostMstar!=satMstar)])
            else:
                if 2 in dips:
                    t_accretion = min(t_Gyr[dips==2]) #Gyr since beginning of simulation
                else:
                    t_accretion = disruption_time(ID=ID, simulation=simulation, status=status, 
                                                  tangos_halo=tangos_halo, halo_id=halo_id, snap_num=snap_num, resolution=resolution)     
        t_accretion = round(t_accretion, 3)
        #id_accretion = np.array(unique_ids)[t_Gyr==t_accretion]
        if category=='stars':
            M_accretion = satMstar[t_Gyr==t_accretion]
            Msat_ratio_accretion = M_accretion/hostMstar[t_Gyr==t_accretion]
            Mhost_ratio_accretion = hostMstar[t_Gyr==t_accretion]/hostMstar[-1]
            # nstar_accretion = satnstar[t_Gyr==t_accretion]
            
        elif category=='dm':
            M_accretion = satMvir[t_Gyr==t_accretion]
            Msat_ratio_accretion = M_accretion/hostMvir[t_Gyr==t_accretion]
            Mhost_ratio_accretion = hostMvir[t_Gyr==t_accretion]/hostMvir[-1]
        else:
            raise ValueError('A very specific bad thing happened. Try category= "stars" or "dm".')
           #0           #1           #2                    #3
    return t_accretion, M_accretion[0], Msat_ratio_accretion[0]#, int(id_accretion[0]), Mhost_ratio_accretion[0]



def max_sSFR_time(ID=0, simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, resolution=1000):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
        status: 'Zombie', 'Survivor': string
        tangos_halo: 0 or a valid Tangos halo object
        halo_id: halo id: 0 or string or numeric
        snap_num: 0 or 4 digit simulation snapshot number: string
    output params:
        maximum sSFR: the maximum sSFR of a dwarf before disruption
        0:  [units: /yr]
        time of maximum sSFR: the earliest time before disruption when a dwarf reaches its maximum sSFR
        1:  [units: Gyr since start of simulation]
    '''
    if ID != 0:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
    np.seterr(divide='ignore')
    #Host
    path = get_file_path(tangos_halo=0, simulation=simulation, status='Host', halo_id='1', snap_num='4096', resolution=resolution)
    with h5py.File(path, 'r') as f:
        t_Gyr = f['time_|_Gyr'][:]
        hostSFR = f['SFR_100Myr_|_Msol/yr'][:]
        hostMstar = f['M_star_|_Msol'][:]
    #hostsSFR = hostSFR/hostMstar
    
    #Satellite
    path = get_file_path(tangos_halo=tangos_halo, simulation=simulation, status=status, halo_id=halo_id, snap_num=snap_num, resolution=resolution)
    with h5py.File(path, 'r') as f:
        satSFR = f['SFR_100Myr_|_Msol/yr'][:]
        satMstar = f['M_star_|_Msol'][:]
    satsSFR = satSFR/satMstar
    satsSFR[satMstar==0] = 0
    
    for i, val in enumerate(satMstar):
        if satMstar[i]==hostMstar[i]:
            satsSFR[i]=-1
        
    if max(satsSFR[satMstar != 0]) > 0:
        max_satsSFR = max(satsSFR[satMstar != 0])
        max_satsSFR_time = min(t_Gyr[satsSFR==max_satsSFR])
    elif max(satSFR[satMstar == 0]) > 0:
        # For the cases where non-zero SFR for zero Mstar, but 0 SFR for all non-zero SFR
        # Take max SFR value with zero Mstar as max sSFR value
        max_satsSFR = max(satSFR[satMstar == 0])
        max_satsSFR_time = min(t_Gyr[satSFR==max_satsSFR])
    else:   
        jump = [t - s for s, t in zip(satMstar, satMstar[1:])]
        max_jump = max(jump)
        max_satsSFR = (max_jump/satMstar[1:][jump==max_jump])[0]
        max_satsSFR_time = min(t_Gyr[1:][jump==max_jump])
    # max_satsSFR value = 1: Halo appears from 0 to X in a snapshot and continues to lose stars
    #     and has 0 SFR throughout
    return max_satsSFR, max_satsSFR_time




def disruption_time(ID=0, simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, resolution=1000):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
        status: 'Zombie', 'Survivor': string
        tangos_halo: 0 or a valid Tangos halo object
        halo_id: halo id: 0 or string or numeric
        snap_num: 0 or 4 digit simulation snapshot number: string
    output params:
        time of disruption: the earliest time when Tangos stops identifying a dwarf as a separate halo from the MPB
        0:  [units: Gyr since start of simulation]
        disruption timescale: returned only if disrupt_timescale=True, the time lag between accretion and disruption
        1:  [units: Gyr]
    note: this is only defined for Zombies. For all other cases, 0, 1 = -1
    '''
    if ID != 0:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
    #Host
    path = get_file_path(tangos_halo=0, simulation=simulation, status='Host', halo_id='1', snap_num='4096', resolution=resolution)
    with h5py.File(path, 'r') as fh:
        t_Gyr = fh['time_|_Gyr'][:]
        hostMstar = fh['M_star_|_Msol'][:]
    #Satellite
    path = get_file_path(tangos_halo=tangos_halo, simulation=simulation, status=status, halo_id=halo_id, snap_num=snap_num, resolution=resolution)
    with h5py.File(path, 'r') as f:
        satMstar = f['M_star_|_Msol'][:]
    if status=='Zombie': 
        if len(satMstar[(satMstar != hostMstar) & (satMstar != 0)]) > 1:
            disruption_time = max(t_Gyr[(satMstar != hostMstar) & (satMstar > 0)])
        else:
            #this is a strange zombie that I want to exclude
            #this zombie did not exist for more than 1 snapshot
            disruption_time = 0 
    else:
        disruption_time = -1 #this is not a zombie so I can't calculate its disruption time
#         disruption_timescale = -1
    return disruption_time#, disruption_timescale



def apocentric_distance(ID=0, simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, resolution=1000):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
        status: 'Zombie', 'Survivor': string
        tangos_halo: 0 or a valid Tangos halo object
        halo_id: halo id: 0 or string or numeric
        snap_num: 0 or 4 digit simulation snapshot number: string
    output params:
        [0] time of closest approach: the earliest time (before disruption) when a satellite gets closest to the host
                [units: Gyrs since start of simulation]
        [1] apocentric distance (distance of closest approach): the minimum distance of separation between the satellite and host CM before disruption
                [units: Virial Radius of Host]
        [2] apocentric distance (distance of closest approach): the minimum distance of separation between the satellite and host CM before disruption
                [units: pkpc]
        
    '''  
    if ID != 0:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
        
    (t, x, y, z, Rvir), (t_new, x_new, y_new, z_new, Rvir_new) = orbit_interpolation(simulation=simulation, status=status, 
                                                                                     tangos_halo=tangos_halo, halo_id=halo_id, 
                                                                                     snap_num=snap_num, resolution=resolution)
    '''
    if list(set(t_new))[0] == 0:
        # Halo doesn't have a long enough track to calculate orbit interpolation
        # Use raw snapshot data
        # Linear distance between CM of host & satellite in pkpc
        radial_distance = np.sqrt(x**2 + y**2 + z**2)
        # Linear distance between CM of host & satellite in Rvir[host]
        normalized_distance = radial_distance/Rvir
        # Distance of closest approach in Rvir[host]
        apocentric_distance = min(normalized_distance[radial_distance!=0])
        # Distance of closest approach in pkpc
        min_lin_distance = min(radial_distance[radial_distance!=0])
        # Time of closest approach
        time_of_closest_approach = min(t[normalized_distance==apocentric_distance])
        
    else:
        # Halo has long enough track to calculate interpolated orbits
        # Use interpolation
        # Linear distance between CM of host & satellite in pkpc
        radial_distance = np.sqrt(x_new**2 + y_new**2 + z_new**2)
        # Linear distance between CM of host & satellite in Rvir[host]
        normalized_distance = radial_distance/Rvir_new
        # Distance of closest approach in Rvir[host]
        apocentric_distance = min(normalized_distance[radial_distance!=0])
        # Distance of closest approach in pkpc
        min_lin_distance = min(radial_distance[radial_distance!=0])
        # Time of closest approach
        time_of_closest_approach = min(t_new[normalized_distance==apocentric_distance])
    '''
    R = np.sqrt(x**2 + y**2 + z**2)                                                                                 
    min_lin_distance = min(R[R!=0])
    apocentric_distance = min_lin_distance/Rvir[R==min_lin_distance]
    time_of_closest_approach = t[R==min_lin_distance]
                                                                                     
    return time_of_closest_approach[0], apocentric_distance[0], min_lin_distance




def orbit_interpolation(ID=0, simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, resolution=1000):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
        status: 'Zombie', 'Survivor': string
        tangos_halo: 0 or a valid Tangos halo object
        halo_id: halo id: 0 or string or numeric
        snap_num: 0 or 4 digit simulation snapshot number: string
    output params:
        apocentric distance (distance of closest approach): the minimum distance of separation between the satellite and host CM before disruption
            [units: Virial Radius of Host]
        time of closest approach: the earliest time (before disruption) when a satellite gets closest to the host
            [units: Gyrs since start of simulation]
    '''
    from scipy.interpolate import CubicSpline
    if ID != 0:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
    #Host
    path = get_file_path(tangos_halo=0, simulation=simulation, status='Host', halo_id='1', snap_num='4096', resolution=resolution)
    with h5py.File(path, 'r') as fh:
        t_Gyr = fh['time_|_Gyr'][:]
        hostx = fh['Xc_|_kpc'][:]
        hosty = fh['Yc_|_kpc'][:]
        hostz = fh['Zc_|_kpc'][:]
        hostRvir = fh['Rvir_|_kpc'][:]
    #Satellite
    path = get_file_path(tangos_halo=tangos_halo, simulation=simulation, status=status, halo_id=halo_id, snap_num=snap_num, resolution=resolution)
    with h5py.File(path, 'r') as f:
        satx = f['Xc_|_kpc'][:]
        saty = f['Yc_|_kpc'][:]
        satz = f['Zc_|_kpc'][:]
        satMstar = f['M_star_|_Msol'][:]
    radial_distance = np.sqrt((hostx-satx)**2 + (hosty-saty)**2 + (hostz-satz)**2)
    t = t_Gyr[satMstar!=0]
    x = (satx-hostx)[satMstar!=0]
    y = (saty-hosty)[satMstar!=0]
    z = (satz-hostz)[satMstar!=0]
    Rvir = hostRvir[satMstar!=0] # Host Rvir
    
    dummy = [0]*3
    if len(t)==0:
        return (dummy, dummy, dummy, dummy, dummy), (dummy, dummy, dummy, dummy, dummy)
    elif len(t)>=2:
        # From: https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
        # time
        t_new = np.linspace(min(t), max(t), 300)
        # relative x coords
        f = CubicSpline(t, x) #fxn
        x_new = f(t_new)
        # relative y coords
        f = CubicSpline(t, y) #fxn
        y_new = f(t_new)
        # relative z coords
        f = CubicSpline(t, z) #fxn
        z_new = f(t_new)
        # host Rvir
        f = CubicSpline(t, Rvir) #fxn
        Rvir_new = f(t_new)
        return (t, x, y, z, Rvir), (t_new, x_new, y_new, z_new, Rvir_new)
    else:
        return (t, x, y, z, Rvir), (dummy, dummy, dummy, dummy, dummy)

    
def quenching_time(ID=0, simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, threshold=2e-11, resolution=1000):
    '''
    input params: 
        ID: new sim/snap/halo id
        simulation: h148, h229, h242, h329: string
        status: 'Zombie', 'Survivor': string
        tangos_halo: 0 or a valid Tangos halo object
        halo_id: halo id: 0 or string or numeric
        snap_num: 0 or 4 digit simulation snapshot number: string
        threshold: threshold in sSFR below which a galaxy is quenched
            [units: /yr]
    output params:
        quenching time: earliest time a satellite falls below the threshold sSFR 
            [units: Gyr since start of simulation]
    '''
    import pandas as pd
    if resolution==1000:
        d = pd.read_csv('sSFR_data.csv')
        ids = d['IDs'].to_numpy()
        sSFRs = d['sSFR'].to_numpy()
        sSFR = sSFRs[ids==ID][0] # A string of form [a b c ...]
        sSFR = np.array(list(map(float, sSFR[1:-1].split())))/1e6
    elif resolution==100:
        d = pd.read_csv('sSFR_data100.csv')
        ids = d['ID'].to_numpy()
        sSFRs = d['sSFR'].to_numpy()
        sSFR = sSFRs[ids==ID][0] # A string of form [a b c ...]
        sSFR = np.array(list(map(float, sSFR[1:-1].split())))
    else:
        raise ValueError("Resolution not implemented yet.")
    
#     Mstar = d['Mass'].to_numpy()
#     new_Mstar = d['']
    
    #np.seterr(divide='ignore')
    if ID != 0:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
        
    #Host
    path = get_file_path(tangos_halo=0, simulation=simulation, status='Host', halo_id='1', snap_num='4096', resolution=resolution)
    with h5py.File(path, 'r') as fh:
        t_Gyr = fh['time_|_Gyr'][:]
    t_Gyr = np.array([round(t, 3) for t in t_Gyr])
    
    #print(sSFR)
    # Replace nan and inf values with 0
    # When making log plots of sSFR
    
    t_infall = accretion_time(ID=ID, simulation=simulation, status=status, tangos_halo=tangos_halo, 
                              halo_id=halo_id, snap_num=snap_num, category='stars', resolution=resolution)[0]
    if status=='Survivor':
        t_final_sSFR = t_Gyr[-1]
    else:
        t_final_sSFR = max(t_Gyr[sSFR>=0])
    final_sSFR = sSFR[t_Gyr==t_final_sSFR]
    if final_sSFR<=threshold and final_sSFR>=0: 
        #quenched at z=0 or final snap before disruption
            #z=0 or final stellar mass is non-zero
        #find times when the sSFR transitioned from above to below the threshold
        dips = np.zeros(len(sSFR))
        for i, val in enumerate(sSFR):
            if i>0 and sSFR[i]<=threshold and sSFR[i-1]>threshold and sSFR[i]>=0:
                #sSFR at current timestep is below threshold
                    #sSFR at previous timestep was above threshold
                        #the satellite has not merged with host
                            #current stellar mass is nonzero
                                #current sSFR is non-negative
                dips[i]=1
        if len(t_Gyr[dips==1]) != 0:
            #the galaxy has transitioned from star-forming to quiescent at least once before z=0
            #recall that these galaxies are quenched at z=0. so these galaxies have always been below the threshold
            #find the latest time that a galaxy crossed from star-forming to quiescent
            quenching_time = max(t_Gyr[dips==1]) 
        else:
            #this survivor has never transitioned from star-forming to quiescent before z=0
            #recall that these galaxies are quenched at z=0
            #find earliest time the galaxy met the sSFR threshold
            # Check how many survivors fall here: many do :((
            quenching_time = min(t_Gyr[(sSFR<=threshold) & (sSFR>=0)])
    else: 
        #this survivor is star-forming at z=0
        quenching_time = -1

#     else:
#         #quenching_time = -1
#         raise valueError('Halo must be a Survivor or Zombie. Try status = "Zombie" or "Survivor".')
    
    return quenching_time, sSFR, t_Gyr

"""    
def quenching_time(ID=0, simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, threshold=1e-11):
    '''
    input params: 
        ID: new sim/snap/halo id
        simulation: h148, h229, h242, h329: string
        status: 'Zombie', 'Survivor': string
        tangos_halo: 0 or a valid Tangos halo object
        halo_id: halo id: 0 or string or numeric
        snap_num: 0 or 4 digit simulation snapshot number: string
        threshold: threshold in sSFR below which a galaxy is quenched
            [units: /yr]
    output params:
        quenching time: earliest time a satellite falls below the threshold sSFR 
            [units: Gyr since start of simulation]
    '''
    #np.seterr(divide='ignore')
    if ID != 0:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
        
    #Host
    path = get_file_path(tangos_halo=0, simulation=simulation, status='Host', halo_id='1', snap_num='4096')
    with h5py.File(path, 'r') as fh:
        t_Gyr = fh['time_|_Gyr'][:]
        hostSFR = fh['SFR_100Myr_|_Msol/yr'][:]
        hostMstar = fh['M_star_|_Msol'][:]
    #hostsSFR = hostSFR/hostMstar 
    
    #Satellite
    path = get_file_path(tangos_halo=tangos_halo, simulation=simulation, status=status, halo_id=halo_id, snap_num=snap_num)
    with h5py.File(path, 'r') as f:
        satSFR = f['SFR_100Myr_|_Msol/yr'][:]
        satMstar = f['M_star_|_Msol'][:]
    satsSFR = satSFR/satMstar
    #replace nan and inf values with 0
    satsSFR[satMstar == 0] = 0
    #satsSFR[satsSFR == 0] = 1e-12 # When making log plots of sSFR
    
    t_infall = accretion_time(ID=ID, simulation=simulation, status=status, tangos_halo=tangos_halo, 
                              halo_id=halo_id, snap_num=snap_num, category='stars')[1]
    if status == 'Survivor':
        if satMstar[-1] == 0:
            #this survivor has no stars at z=0, so we need to exclude this
            quenching_time = -2
        elif satsSFR[-1]<=threshold and satsSFR[-1]>=0 and satMstar[-1]>0: 
            #quenched at z=0
                #z=0 stellar mass is non-zero
            #find times when the sSFR transitioned from above to below the threshold
            dips = np.zeros(len(satsSFR))
            for i, val in enumerate(satsSFR):
                if i>0 and satsSFR[i]<=threshold and satsSFR[i-1]>threshold and hostMstar[i]!=satMstar[i] and satMstar[i]>0 and satsSFR[i]>=0:
                    #sSFR at current timestep is below threshold
                        #sSFR at previous timestep was above threshold
                            #the satellite has not merged with host
                                #current stellar mass is nonzero
                                    #current sSFR is non-negative
                    dips[i]=1
            if len(t_Gyr[dips==1]) != 0:
                #the galaxy has transitioned from star-forming to quiescent at least once before z=0
                #recall that these galaxies are quenched at z=0. so these galaxies have always been below the threshold
                #find the latest time that a galaxy crossed from star-forming to quiescent
                quenching_time = max(t_Gyr[dips==1]) 
            else:
                #this survivor has never transitioned from star-forming to quiescent before z=0
                #recall that these galaxies are quenched at z=0
                #find earliest time the galaxy met the sSFR threshold
                # Check how many survivors fall here: many do :((
                quenching_time = min(t_Gyr[(satsSFR<=threshold) & (hostMstar!=satMstar) & (satMstar!=0) & (satsSFR>=0)])
                
        else: 
            #this survivor is star-forming at z=0
            quenching_time = -1
            
    elif status == 'Zombie': #disrupted before z=0
        t_disruption = disruption_time(simulation=simulation, status=status, tangos_halo=tangos_halo, 
                                       halo_id=halo_id, snap_num=snap_num)
        if t_disruption == 0: #or satMstar[t_Gyr==t_infall] == 0 or satMstar[-1] == 0:
            #this is one of the weird zombies that has stars for only one snapshot, identified by disruption_time()=0 
            #I want my zombies to have some stars for at least 2 snapshots, and at infall
            quenching_time = -2
        elif satsSFR[t_Gyr == t_disruption]<=threshold and satMstar[t_Gyr == t_infall] != 0: 
            #quenched at/before disruption
            #find times when the sSFR transitioned from above to below the threshold
            dips = np.zeros(len(satsSFR))
            for i, val in enumerate(satsSFR):
                if i > 0 and satsSFR[i]<=threshold and satsSFR[i-1]>threshold and hostMstar[i]!=satMstar[i] and satMstar[i]!=0: 
                    #sSFR at current timestep is below threshold
                        #sSFR at previous timestep was above threshold
                            #the satellite has not merged with host
                                #current stellar mass is nonzero
                    dips[i]=1
                    
            if len(t_Gyr[dips==1]) != 0:
                #the galaxy has transitioned from star-forming to quiescent at least once before disruption
                #find the latest time that a galaxy crossed from star-forming to quiescent
                quenching_time = max(t_Gyr[dips==1]) 
            else:
                #the galaxy has never transitioned from star-forming to quiescent before disruption
                #recall that these galaxies are quenched at disruption
                #find earliest time the galaxy met the sSFR threshold
                quenching_time = min(t_Gyr[(satsSFR<=threshold) & (satMstar!=0) & (satsSFR>=0)])

        else: 
            #this zombie was star-forming at disruption
            quenching_time = -1
    else:
        #quenching_time = -1
        raise valueError('Halo must be a Survivor or Zombie. Try status = "Zombie" or "Survivor".')
    
    return quenching_time#, satsSFR[satMstar != hostMstar], t_Gyr[satMstar != hostMstar]-t_infall, satMstar[satMstar != hostMstar]
"""


def infall_velocity(ID=0, simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, resolution=1000):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
        key: pre-existing tangos halo property: string
        tangos_halo: 0 or a valid Tangos halo object
        halo_path: 0 or a complete tangos halo address: path or string object
        halo_id: halo id: 0 or string or numeric
        snap_num: 0 or 4 digit simulation snapshot number: string
        category: 'stars' or 'dm': string
    output params:
        relative velocity at infall: magnitude of host-centered relative velocity at infall
            [units: km/s]
        radial velocity at infall: magnitude of host-centered relative radial component of velocity at in fall
            [units: km/s] [negative values point inward towards the CM of the host]
        tangential velocity at infall: magnitude of host-centered relative tangential component of velocity at infall
            [units: km/s]
        velocity at infall: magnitude of un-centered velocity at infall
            [units: km/s]
    '''
    if ID != 0:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
    #Host
    path = get_file_path(tangos_halo=0, simulation=simulation, status='Host', halo_id='1', snap_num='4096', resolution=resolution)
    with h5py.File(path, 'r') as fh:
        t_Gyr = fh['time_|_Gyr'][:]
        hostMstar = fh['M_star_|_Msol'][:]
        hostVXc = fh['VXc_|_km/s'][:]
        hostVYc = fh['VYc_|_km/s'][:]
        hostVZc = fh['VZc_|_km/s'][:]
        hostx = fh['Xc_|_kpc'][:]
        hosty = fh['Yc_|_kpc'][:]
        hostz = fh['Zc_|_kpc'][:]
    t_Gyr = np.array([round(t, 3) for t in t_Gyr])
    #Satellite
    path = get_file_path(tangos_halo=tangos_halo, simulation=simulation, 
                         status=status, halo_id=halo_id, snap_num=snap_num, resolution=resolution)
    with h5py.File(path, 'r') as f:
        satMstar = f['M_star_|_Msol'][:]
        satVXc = f['VXc_|_km/s'][:]
        satVYc = f['VYc_|_km/s'][:]
        satVZc = f['VZc_|_km/s'][:]
        satx = f['Xc_|_kpc'][:]
        saty = f['Yc_|_kpc'][:]
        satz = f['Zc_|_kpc'][:]
    t_accretion = accretion_time(ID=ID, simulation=simulation, status=status, 
                                 tangos_halo=tangos_halo, halo_id=halo_id, 
                                 snap_num=snap_num, category='stars', resolution=resolution)[0]
    #|v_{sat}|
    velocity = (satVXc**2 + satVYc**2 + satVZc**2)**(1/2)
    #|v_{sat} - v_{host}|
    relative_velocity = ((satVXc - hostVXc)**2 + (satVYc - hostVYc)**2 + (satVZc - hostVZc)**2)**(1/2)
    #|v_{rad}|
    radial_velocity = ((satVXc - hostVXc)*(satx - hostx) + 
                       (satVYc - hostVYc)*(saty - hosty) + 
                       (satVZc - hostVZc)*(satz - hostz))/((satx - hostx)**2 + 
                                                           (saty - hosty)**2 + 
                                                           (satz - hostz)**2)**(1/2)
    #|v_{tan}|
    tangential_velocity = (relative_velocity**2 - radial_velocity**2)**(1/2)
    
    #velocities at infall
    infall_relative_velocity = relative_velocity[t_Gyr == t_accretion]
    infall_radial_velocity = radial_velocity[t_Gyr == t_accretion]
    infall_tangential_velocity = tangential_velocity[t_Gyr == t_accretion]
    infall_velocity = velocity[t_Gyr == t_accretion]
    return infall_relative_velocity[0], infall_radial_velocity[0], infall_tangential_velocity[0], infall_velocity[0]



def infall_final_coordinates(ID=0, simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, resolution=1000):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
        status: 'Zombie', 'Survivor': string
        tangos_halo: 0 or a valid Tangos halo object
        halo_id: halo id: 0 or string or numeric
        snap_num: 0 or 4 digit simulation snapshot number: string
    output params:
        [0] coordinates at infall: Relative x, [pkpc]
                                            y, [pkpc]
                                            z, [pkpc]
                                            linear distance, [pkpc]
                                            angle and [radian]
                                            distance away from z axis of satellite CM from host CM [pkpc]
        [1] coordinates at final snapshot: Relative x, [pkpc]
                                                    y, [pkpc]
                                                    z, [pkpc]
                                                    linear distance, [pkpc]
                                                    angle and [radian]
                                                    distance away from z axis of satellite CM from host CM [pkpc]     
    '''  
    if ID != 0:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
    # Host
    path = get_file_path(tangos_halo=0, simulation=simulation, status='Host', halo_id='1', snap_num='4096', resolution=resolution)
    with h5py.File(path, 'r') as fh:
        t_Gyr = fh['time_|_Gyr'][:]
        hostMstar = fh['M_star_|_Msol'][:]
        hostx = fh['Xc_|_kpc'][:]
        hosty = fh['Yc_|_kpc'][:]
        hostz = fh['Zc_|_kpc'][:]
    t_Gyr = np.array([round(t, 3) for t in t_Gyr])
    # Satellite
    path = get_file_path(tangos_halo=tangos_halo, simulation=simulation, 
                         status=status, halo_id=halo_id, snap_num=snap_num, resolution=resolution)
    with h5py.File(path, 'r') as f:
        satMstar = f['M_star_|_Msol'][:]
        satx = f['Xc_|_kpc'][:]
        saty = f['Yc_|_kpc'][:]
        satz = f['Zc_|_kpc'][:]
    t_accretion = accretion_time(ID=ID, simulation=simulation, status=status, 
                                 tangos_halo=tangos_halo, halo_id=halo_id, 
                                 snap_num=snap_num, category='stars', resolution=resolution)[0]
    # At Infall:
    x_i = (satx-hostx)[t_Gyr == t_accretion][0] #pkpc
    y_i = (saty-hosty)[t_Gyr == t_accretion][0] #pkpc
    z_i = (satz-hostz)[t_Gyr == t_accretion][0] #pkpc
    R_i = np.sqrt(x_i**2 + y_i**2 + z_i**2) #pkpc
    cosPhi_i = z_i/R_i
    Phi_i = np.arccos(cosPhi_i) #Radians
    z_i_away = z_i*np.tan(Phi_i) #pkpc: perpendicular distance away from z-axis
    
    # At final snapshot:
    t_final = max(t_Gyr[(hostMstar != satMstar) & (satMstar>0)])
    x_f = (satx-hostx)[t_Gyr == t_final][0] #pkpc
    y_f = (saty-hosty)[t_Gyr == t_final][0] #pkpc
    z_f = (satz-hostz)[t_Gyr == t_final][0] #pkpc
    R_f = np.sqrt(x_f**2 + y_f**2 + z_f**2) #pkpc 
    cosPhi_f = z_f/R_f
    Phi_f = np.arccos(cosPhi_f) #Radians
    z_f_away = z_f*np.tan(Phi_f) #pkpc: perpendicular distance away from z-axis
    
    return (x_i, y_i, z_i, R_i, Phi_i, z_i_away), (x_f, y_f, z_f, R_f, Phi_f, z_f_away)




def infall_final_n_particles(ID=0, simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, resolution=1000):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
        status: 'Zombie', 'Survivor': string
        tangos_halo: 0 or a valid Tangos halo object
        halo_id: halo id: 0 or string or numeric
        snap_num: 0 or 4 digit simulation snapshot number: string
    output params:
        [0] number of <> particles at infall: Number of gas, star, and dark matter particles of satellite
                [units: none]
        [1] number of <> particles at final snapshot: Number of gas, star, and dark matter particles of satellite
                [units: none]     
    '''  
    if ID != 0:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
    # Host
    path = get_file_path(tangos_halo=0, simulation=simulation, status='Host', halo_id='1', snap_num='4096', resolution=resolution)
    with h5py.File(path, 'r') as fh:
        t_Gyr = fh['time_|_Gyr'][:]
        hostMstar = fh['M_star_|_Msol'][:]
    t_Gyr = np.array([round(t, 3) for t in t_Gyr])
    # Satellite
    path = get_file_path(tangos_halo=tangos_halo, simulation=simulation, 
                         status=status, halo_id=halo_id, snap_num=snap_num, resolution=resolution)
    with h5py.File(path, 'r') as f:
        satMstar = f['M_star_|_Msol'][:]
        satn_gas = f['n_gas'][:]
        satn_star = f['n_star'][:]
        satn_dm = f['n_dm'][:]
    t_accretion = accretion_time(ID=ID, simulation=simulation, status=status, 
                                 tangos_halo=tangos_halo, halo_id=halo_id, 
                                 snap_num=snap_num, category='stars', resolution=resolution)[0]
    # At Infall:
    n_gas_i = satn_gas[t_Gyr == t_accretion][0] #pkpc
    n_star_i = satn_star[t_Gyr == t_accretion][0] #pkpc
    n_dm_i = satn_dm[t_Gyr == t_accretion][0] #pkpc
    
    # At final snapshot:
    t_final = max(t_Gyr[hostMstar != satMstar])
    n_gas_f = satn_gas[t_Gyr == t_final][0] #pkpc
    n_star_f = satn_star[t_Gyr == t_final][0] #pkpc
    n_dm_f = satn_dm[t_Gyr == t_final][0] #pkpc
      
    return (n_gas_i, n_star_i, n_dm_i), (n_gas_f, n_star_f, n_dm_f)



def impact_parameter(ID=0, simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, resolution=1000):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
        status: 'Zombie', 'Survivor': string
        tangos_halo: 0 or a valid Tangos halo object
        halo_id: halo id: 0 or string or numeric
        snap_num: 0 or 4 digit simulation snapshot number: string
    output params:
        First local minima in R after infall
        [0] impact parameter: apocentric distance (distance of closest approach) of satellite CM from host CM after first infall
                [units: kpc]
        [1] impact parameter: apocentric distance (distance of closest approach) of satellite CM from host CM after first infall
                [units: host Rvir]
        [2] time of closest approach after first infall: 
                        positive values: satellite completed at least one approach (fell in & went out at least once)
                        negative values: satellite did not complete one approach (fell in for first time)
                [units: Gyr since start of simulation]
    '''  
    if ID != 0:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
    #Host
    path = get_file_path(tangos_halo=0, simulation=simulation, status='Host', halo_id='1', snap_num='4096', resolution=resolution)
    with h5py.File(path, 'r') as fh:
        t_Gyr = fh['time_|_Gyr'][:]
        hostx = fh['Xc_|_kpc'][:]
        hosty = fh['Yc_|_kpc'][:]
        hostz = fh['Zc_|_kpc'][:]
        hostMvir = fh['Mvir_|_Msol'][:]
        hostRvir = fh['Rvir_|_kpc'][:]
        hostMstar = fh['M_star_|_Msol'][:]
    t_Gyr = np.array([round(t, 3) for t in t_Gyr])
    #Satellite
    path = get_file_path(tangos_halo=tangos_halo, simulation=simulation, status=status, halo_id=halo_id, snap_num=snap_num, resolution=resolution)
    with h5py.File(path, 'r') as f:
        satx = f['Xc_|_kpc'][:]
        saty = f['Yc_|_kpc'][:]
        satz = f['Zc_|_kpc'][:]
        satMvir = f['Mvir_|_Msol'][:]
        satMstar = f['M_star_|_Msol'][:]
        
    R = ((hostx-satx)**2 + (hosty-saty)**2 + (hostz-satz)**2)**(1/2) #pkpc
    R_hostRvir = R/hostRvir
    # First time satellite CM crosses into 
    t_infall = accretion_time(ID=ID, simulation=simulation, status=status, tangos_halo=tangos_halo, 
                              halo_id=halo_id, snap_num=snap_num, category='stars', resolution=resolution)[0]
    minima = np.zeros(len(R))
    for i, val in enumerate(R[:-1]):
        if i > 0 and t_Gyr[i]>=t_infall and R[i]<=R[i+1] and R[i]<=R[i-1] and R[i]!=0:
            minima[i]=1  
    if 1 in minima:
        # First local minima in position of satellite CM from host CM
        t_impact = min(t_Gyr[minima==1]) #Gyr since beginning of simulation
        d_impact = R[t_Gyr==t_impact][0] #impact parameter after first pass in pkpc
        d_impact_Rvir = d_impact/hostRvir[R==d_impact][0] #impact parameter after first pass in host Rvir
    else:
        d_impact = min(R[(R!=0) & (satMstar>0)])
        d_impact_Rvir = d_impact/hostRvir[R==d_impact][0]
        t_impact = -t_Gyr[R==d_impact][0]
                                                                                           
    return d_impact, d_impact_Rvir, t_impact



def velocity_angles(ID=0, simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, resolution=1000):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
        key: pre-existing tangos halo property: string
        tangos_halo: 0 or a valid Tangos halo object
        halo_path: 0 or a complete tangos halo address: path or string object
        halo_id: halo id: 0 or string or numeric
        snap_num: 0 or 4 digit simulation snapshot number: string
    output params:
        
    '''
    from scipy.interpolate import CubicSpline
    np.seterr(divide='ignore', invalid='ignore')
    if ID != 0:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
    #Host
    path = get_file_path(tangos_halo=0, simulation=simulation, status='Host', halo_id='1', snap_num='4096', resolution=resolution)
    with h5py.File(path, 'r') as fh:
        t_Gyr = fh['time_|_Gyr'][:]
        hostMstar = fh['M_star_|_Msol'][:]
        hostVX = fh['VXc_|_km/s'][:]
        hostVY = fh['VYc_|_km/s'][:]
        hostVZ = fh['VZc_|_km/s'][:]
        hostx = fh['Xc_|_kpc'][:]
        hosty = fh['Yc_|_kpc'][:]
        hostz = fh['Zc_|_kpc'][:]
    t_Gyr = np.array([round(t, 3) for t in t_Gyr])
    #Satellite
    path = get_file_path(tangos_halo=tangos_halo, simulation=simulation, 
                         status=status, halo_id=halo_id, snap_num=snap_num, resolution=resolution)
    with h5py.File(path, 'r') as f:
        satMstar = f['M_star_|_Msol'][:]
        satVX = f['VXc_|_km/s'][:]
        satVY = f['VYc_|_km/s'][:]
        satVZ = f['VZc_|_km/s'][:]
        satx = f['Xc_|_kpc'][:]
        saty = f['Yc_|_kpc'][:]
        satz = f['Zc_|_kpc'][:]
    t_accretion = accretion_time(ID=ID, simulation=simulation, status=status, 
                                 tangos_halo=tangos_halo, halo_id=halo_id, 
                                 snap_num=snap_num, category='stars', resolution=resolution)[0]

    
    
    # time
    t = np.linspace(min(t_Gyr[satMstar!=hostMstar]), max(t_Gyr[satMstar!=hostMstar]), 300)
    # sat x coords
    f = CubicSpline(t_Gyr[satMstar!=hostMstar], satx[satMstar!=hostMstar]) #fxn
    satx = f(t)
    # sat y coords
    f = CubicSpline(t_Gyr[satMstar!=hostMstar], saty[satMstar!=hostMstar]) #fxn
    saty = f(t)
    # sat z coords
    f = CubicSpline(t_Gyr[satMstar!=hostMstar], satz[satMstar!=hostMstar]) #fxn
    satz = f(t)
    # host x coords
    f = CubicSpline(t_Gyr[satMstar!=hostMstar], hostx[satMstar!=hostMstar]) #fxn
    hostx = f(t)
    # host y coords
    f = CubicSpline(t_Gyr[satMstar!=hostMstar], hosty[satMstar!=hostMstar]) #fxn
    hosty = f(t)
    # host z coords
    f = CubicSpline(t_Gyr[satMstar!=hostMstar], hostz[satMstar!=hostMstar]) #fxn
    hostz = f(t)
    # host x cm velocity
    f = CubicSpline(t_Gyr[satMstar!=hostMstar], hostVX[satMstar!=hostMstar]) #fxn
    hostVX = f(t)
    # host y cm velocity
    f = CubicSpline(t_Gyr[satMstar!=hostMstar], hostVY[satMstar!=hostMstar]) #fxn
    hostVY = f(t)
    # host z cm velocity
    f = CubicSpline(t_Gyr[satMstar!=hostMstar], hostVZ[satMstar!=hostMstar]) #fxn
    hostVZ = f(t)
    # sat x cm velocity
    f = CubicSpline(t_Gyr[satMstar!=hostMstar], satVX[satMstar!=hostMstar]) #fxn
    satVX = f(t)
    # sat y cm velocity
    f = CubicSpline(t_Gyr[satMstar!=hostMstar], satVY[satMstar!=hostMstar]) #fxn
    satVY = f(t)
    # sat z cm velocity
    f = CubicSpline(t_Gyr[satMstar!=hostMstar], satVZ[satMstar!=hostMstar]) #fxn
    satVZ = f(t)
    
    #|v_{sat}|
    velocity = (satVX**2 + satVY**2 + satVZ**2)**(1/2)
    #|v_{sat} - v_{host}|
    Vx = satVX - hostVX
    Vy = satVY - hostVY
    Vz = satVZ - hostVZ
    X = satx - hostx
    Y = saty - hosty
    Z = satz - hostz
    R = (X**2 + Y**2 + Z**2)**(1/2)
    
    cosTheta = Z/R # Vertical angle away from z
    Theta = np.arccos(cosTheta) # in Radians
    m = R/np.sin(Theta)
    
    tanPhi = Y/X
    Phi = np.arctan(tanPhi)
    XTheta = m*np.cos(Phi)-satx
    YTheta = m*np.sin(Phi)-saty
    ZTheta = -satz
    
    XPhi = Z*np.tan(Theta)/np.cos(Phi)
    YPhi = -hosty
    ZPhi = Z
    
    vRelative = (Vx**2 + Vy**2 + Vz**2)**(1/2)
    vRadial = abs((Vx*X + Vy*Y + Vz*Z)/(X**2 + Y**2 + Z**2)**(1/2)) #v.r/|r|  #|v_{rad}|
    vTangential = (vRelative**2 - vRadial**2)**(1/2) #|v_{tan}|
    
    vTheta = abs((Vx*XTheta + Vy*YTheta + Vz*ZTheta)/(XTheta**2 + YTheta**2 + ZTheta**2)**(1/2))
    vPhi = abs((Vx*XPhi + Vy*YPhi + Vz*ZPhi)/(XPhi**2 + YPhi**2 + ZPhi**2)**(1/2))
    
    #velocities at infall
    return vTheta, vPhi, vRadial, t




def get_main_progenitor_branch(tangos_halo, simulation, resolution=1000):
    '''
    input params: 
        tangos_halo: a valid Tangos halo object
        simulation: h148, h229, h242, h329: string
    output params:
        Main Progenitor Branch: 
                list of main progenitor branch halos from earliest to latest snapshot
        ids of Main Progenitor Branch:
                list of main progenitor branch halo ids from earliest to latest snapshot
    '''
    if resolution==1000:
        tangos.init_db("simulation_database/"+ str(simulation) +".db")
    elif resolution==100:
        tangos.init_db("/data/Sims/Databases/100Nptcls/"+ str(simulation) +".db")
    else:
        raise ValueError("Resolution not implemented yet.")
    ids = track_halo_property(simulation=simulation, key="halo_number()", 
                              tangos_halo=tangos_halo, halo_path=0, 
                              halo_id=0, snap_num=0, resolution=resolution) # earliest to latest timestep
    MPB = np.zeros(0)
    unique_ids=[] # earliest to most recent timestestep
    x = 0
    timesteps = tangos.get_simulation("snapshots").timesteps
    snapnums = get_timesteps(simulation=simulation, resolution=resolution)[3]
    while x <= len(timesteps)-1:
        x = int(x)
        # idx get_host= int(ids_flip[x]) 
        idx = int(ids[x])
        if idx == 0:
            MPB = np.concatenate((MPB, [-1]), axis=None)
            unique_ids.append(0)
        else:
            idk=str(simulation)[1:]+str(int(snapnums[x]))+str(idx)
            MPB = np.concatenate((MPB, [timesteps[x][idx]]), axis=None)
            unique_ids.append(idk)
        x += 1
    return MPB, unique_ids


def near_mint_disruption_time(ID=0, num_ptcls=800, simulation=0, status=0, tangos_halo=0, halo_id=0, snap_num=0, resolution='Mint'):
    '''
    input params: 
        simulation: h148, h229, h242, h329: string
        status: 'Zombie', 'Survivor': string
        tangos_halo: 0 or a valid Tangos halo object
        halo_id: halo id: 0 or string or numeric
        snap_num: 0 or 4 digit simulation snapshot number: string
        category: 'stars' or 'dm': string
    output params:
        time of disruption: 
                [units: Gyr since start of simulation]
    '''
    if ID != 0:
        simulation, status, halo_id, snap_num = ID_to_sim_halo_snap(ID=ID)
    #Host
    path = get_file_path(tangos_halo=0, simulation=simulation, status='Host', halo_id='1', snap_num='4096', resolution=resolution)
    with h5py.File(path, 'r') as fh:
        t_Gyr = fh['time_|_Gyr'][:]
        hostMvir = fh['Mvir_|_Msol'][:]
        
    #Satellite
    path = get_file_path(tangos_halo=tangos_halo, simulation=simulation, status=status, halo_id=halo_id, snap_num=snap_num, resolution=resolution)
    with h5py.File(path, 'r') as f:
        satMvir = f['Mvir_|_Msol'][:]
        satn_dm = f['n_dm'][:]
        satn_star = f['n_star'][:]
        satn_gas = f['n_gas'][:]
        
    n_ptcls = satn_dm + satn_star + satn_gas
    satn_ptcls = n_ptcls[(hostMvir!=satMvir) & (satMvir>0)]
#     print(satn_ptcls)
    t_Gyr = t_Gyr[(hostMvir!=satMvir) & (satMvir>0)]
    surviving_times = t_Gyr[satn_ptcls >= num_ptcls]
#     print(surviving_times)
    
    t_infall = accretion_time(ID=ID, simulation=simulation, status=status, tangos_halo=tangos_halo, 
                              halo_id=halo_id, snap_num=snap_num, category='stars', resolution=resolution)[0]
    
    if len(surviving_times)<2:
        t_disruption = 0 #Never exceeded near-mint mass resolution for more than 2 ts
        new_status = 'Unborn'
    elif surviving_times[-1]>=13.8:
        t_disruption = -1
        new_status = 'Survivor'
    elif len(surviving_times[surviving_times>=t_infall-0.5])<1:
        t_disruption = max(surviving_times)
        new_status = 'Blimp'
    else:
        t_disruption = max(surviving_times)
        new_status = 'Zombie'
    return t_disruption, new_status
