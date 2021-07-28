import tangos
import pynbody
import numpy as np
import h5py

from .halos import tangos_to_pynbody_halo

def generate_halo_image(tangos_halo, simulation, view='faceon'):
    '''
    input params: 
        tangos_halo: a valid Tangos halo object
        simulation: h148, h229, h242, h329: string
        view: faceon or sideon
    output params:
        a rendered halo image:: param: stellar mass density
    '''
    import matplotlib.pylab as plt
    import pynbody.plot.sph as sph    
    
    tangos.init_db("simulation_database/"+ simulation+".db")
    pyn_halo = tangos_to_pynbody_halo(tangos_halo=tangos_halo, simulation=simulation)
    
    if view=='faceon':
        pynbody.analysis.angmom.faceon(pyn_halo)
    elif view=='sideon':
        pynbody.analysis.angmom.sideon(pyn_halo)

    plt.figure(figsize=(15, 15))

    im = sph.image(pyn_halo.s, qty="rho", units="g cm^-2", ret_im=True, resolution=10000, log=True,
                   width=20, cmap="bone", show_cbar=False, fill_nan=True, fill_val=1e-6, vmin=1e-6, vmax=1e5)#, vmin=0.2, vmax=1.5)
    cb = plt.colorbar(im, pad=0.05)
    cb.set_label(label=r'Stellar Mass Density ($g cm^{-2}$)', fontsize=20)
    cb.ax.tick_params(labelsize=15)
    plt.xlabel('x (kpc)', fontsize=20)
    plt.ylabel('y (kpc)', fontsize=20)
    plt.title(str(simulation), fontsize=24)
    plt.text(x=50, y=75, s='z='+str(round(tangos_halo.timestep.redshift, 4)), fontsize=20, color='white')
    plt.tick_params(which='major', length=10, color='grey')
    plt.tick_params(direction='in', which='both', labelsize=15, bottom=True, top=True, left=True, right=True)
    return
