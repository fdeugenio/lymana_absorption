# coding: utf-8
import os
import pickle
import time
import tqdm

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, griddata

from prospect.sources import CSPSpecBasis, FastStepBasis
from ..lymana_absorption.lymana_optical_depth import tau_IGM



def create_interpolator_grid_file(
    output_file_name,
    wgrid=None, zgrid=None, rgrid=None, xgrid=None,
    ):
    """
    wgrid : 1-d array, optional
        Array of rest-frame source wavelengths.
    zgrid : 1-d array, optional
        Array of source redshifts (I think z>=4.5)
    rgrid : 1-d array, optional
        Array of radii of the ionised bubble; best in log space
    xgrid : 1-d array, optional
        Array of global neutral fractions (0 <= xgrid <= 1)
    """

    sps = CSPSpecBasis(zcontinuous=True, compute_vega_mags=False)

    rgrid = np.logspace(-1, 2, 30) if rgrid is None else rgrid
    xgrid = np.linspace(0., 1., 30) if xgrid is None else xgrid
    zgrid = np.linspace(4.5, 20., 100) if zgrid is None else zgrid
    wgrid_sps = np.copy(sps.wavelengths)
    if wgrid is None:
        wgrid = np.hstack((
            wgrid_sps[wgrid_sps<3500.], wgrid_sps[wgrid_sps>=3500][::25],
            wgrid_sps[-1]))
    else:
        assert (wgrid[0 ]<=wgrid_sps[ 0]), f'`wgrid` must cover FSPS wave grid'
        assert (wgrid[-1]>=wgrid_sps[-1]), f'`wgrid` must cover FSPS wave grid'
        if not np.isclose(wgrid[ 0], wgrid_sps[ 0]):
            warnings.warn(f'{wgrid[0]=}<{wgrid_sps[0]=} useless for prospector')
        if not np.isclose(wgrid[-1], wgrid_sps[-1]):
            warnings.warn(f'{wgrid[-1]=}>{wgrid_sps[-1]=} useless for prospector')

    tau_IGM = np.array([[[
        tau_IGM(wgrid*(1+z), z, R_ion=R_ion, x_HI_global=x_HI)
                    for z in tqdm.tqdm(zgrid, leave=False, colour='red')]
                for R_ion in tqdm.tqdm(rgrid, leave=False, colour='white')]
            for x_HI in tqdm.tqdm(xgrid, colour='green')]
    )

    output = {
         'wgrid': wgrid, 'zgrid': zgrid, 'rgrid': rgrid, 'xgrid': xgrid,
         'tau_igm': tau_IGM}
    with open(output_file_name, 'wb') as fhandle:
        pickle.dump(output, fhandle)
        print(f'tau_igm grid written to {output_file_name=}')

def create_interpolator_grid_file_const_quad(
    output_file_name="grid_IGM_conqu_v0.0.pckl",
    zgrid=None, rgrid=None, xgrid=None,
    deltalam_AA=1.
    ):

    sps = CSPSpecBasis(zcontinuous=True, compute_vega_mags=False)
    wgrid_sps = np.copy(sps.wavelengths)

    # These are the start, "pivot", and end of the wavelength grid.
    # The pivot is where the wavelength binning goes from Δλ=deltalam_AA
    # constant to Δλ/λ^2 = const
    w0, w1, w2 = wgrid_sps[0], 1500., wgrid_sps[-1]

    wgrid_blu = np.arange(w0, w1+deltalam_AA, deltalam_AA)
    dlAA_olam2 = deltalam_AA/w1**2 # Just to save space.
    # Could do np.arange(1./w1, 1./w2, -dlAA_olam2) to have exactly `w1` as a node,
    # but then it's hard to get a node on w2 (b/c dlAA_olam2 is so large it misses w2).
    # Attempting to get both w1 and w2 in the grid would involve choosing N, such that
    # Δλ/λ^N = const and both w1 and w2 are in the grid. This is boring and I have to
    # time for it.
    wgrid_red = np.arange(1./w2, 1./w1, dlAA_olam2)[::-1]
    wgrid_red = 1. / wgrid_red
    wgrid = np.hstack((wgrid_blu, wgrid_red))

    print(f'Calculating {wgrid[[0, -1]]=} AA with {len(wgrid)=} elements')

    create_interpolator_grid_file(
        output_file_name,
        wgrid=wgrid, zgrid=zgrid, rgrid=rgrid, xgrid=xgrid)



class tau_igm_interp():

    def __init__(self, wgrid, zgrid, rgrid, xgrid, interpolator):
        self.w = wgrid
        self.z = zgrid
        self.r = rgrid
        self.x = xgrid
        self.tau_igm = interpolator

    def __call__(self, wl_emit, z_s, R_ion, x_HI_global):
        assert (wl_emit[ 0]>=self.w[ 0]), f'Minimum valid wave is {self.w[0]=} AA'
        assert (wl_emit[-1]<=self.w[-1]), f'Maximum valid wave is {self.w[-1]=} AA'
        assert self.z[0]<=z_s<=self.z[-1], f'z_s must be {self.z[0]}<z_s<{self.z[-1]}'
        assert self.r[0]<=R_ion<=self.r[-1], f'R_ion must be {self.r[0]}<R_ion<{self.r[-1]}'
        assert self.x[0]<=x_HI_global<=self.x[-1], f'x_HI must be {self.x[0]}<x_HI<{self.x[-1]}'
     
        tau_wl = self.tau_igm(x_HI_global, np.log(R_ion), np.log(z_s))
     
        return np.interp(wl_emit, self.w, tau_wl)



def create_interpolator_from_grid_file(
    input_file='grid_IGM.npy', output_file='tau_igm_interp_v0.2.pckl'):

    assert os.path.isfile(input_file), f'Cannot find input grid {input_file=}'
    assert not os.path.isfile(output_file), f'File {output_file=} already exists'
    tau_IGM = np.load(input_file)

    sps = CSPSpecBasis(zcontinuous=True, compute_vega_mags=False)

    rgrid = np.logspace(-1, 2, 30)
    xgrid = np.linspace(0., 1., 30)
    zgrid = np.linspace(4.5, 20., 100)
    wgridold = np.copy(sps.wavelengths)
    wgrid = np.hstack((wgridold[wgridold<3500.], wgridold[wgridold>=3500][::25], wgridold[-1]))


    coords = np.meshgrid(xgrid, rgrid, zgrid, indexing='ij')
    coords = np.vstack([
         # x_HI_global   ,    log R_ion             ,    log z
        coords[0].ravel(), np.log(coords[1].ravel()), np.log(coords[2].ravel())
        ]).T
    _tau_igm_wgrid_interp_ = LinearNDInterpolator(
        coords, tau_IGM.reshape(-1, wgrid.size)
        )

    """
    def tau_igm_interp(wl_emit, z_s, R_ion, x_HI_global):
        assert (wl_emit[ 0]>=wgrid[ 0]), f'Minimum valid wave is {wgrid[0]=} AA'
        assert (wl_emit[-1]<=wgrid[-1]), f'Maximum valid wave is {wgrid[-1]=} AA'
        assert zgrid[0]<=z_s<=zgrid[-1], f'z_s must be {zgrid[0]}<z_s<{zgrid[-1]}'
        assert rgrid[0]<=R_ion<=rgrid[-1], f'R_ion must be {rgrid[0]}<R_ion<{rgrid[-1]}'
        assert xgrid[0]<=x_HI<=xgrid[-1], f'x_HI must be {xgrid[0]}<x_HI<{xgrid[-1]}'
     
        tau_wl = _tau_igm_wgrid_interp_(x_HI_global, np.log(R_ion), np.log(z_s))
        
        return np.interp(wl_emit, wgrid, tau_wl)
    """

    _tau_igm_interp_object_ = tau_igm_interp(
        wgrid, zgrid, rgrid, xgrid, _tau_igm_wgrid_interp_)
    output = {
         'wgrid': wgrid, 'zgrid': zgrid, 'rgrid': rgrid, 'xgrid': xgrid,
         'tau_igm_interp': _tau_igm_interp_object_}

    with open(output_file, 'wb') as fhandle:
        pickle.dump(output, fhandle)
        print(f'Successfully written to file {output_file}')



def test_grid(n_tests=10, input_file='tau_igm_interp_v0.2.pckl'):

    interpolator = pickle.load(open(input_file, 'rb'))
    wgrid, zgrid, rgrid, xgrid = [
        interpolator[key] for key in ('wgrid', 'zgrid', 'rgrid', 'xgrid')]
    tau_igm_inter = interpolator['tau_igm_interp']

    np.random.seed(8614)
    
    z_array = sorted(np.random.uniform(zgrid[0], zgrid[-1], size=n_tests))
    cmap = matplotlib.colormaps['nipy_spectral']
    cmap = matplotlib.colormaps['viridis']
    norm = matplotlib.colors.Normalize(vmin=zgrid[0], vmax=zgrid[-1])

    fig, axes = plt.subplots(1, 1, figsize=(20, 12))

    for i in range(n_tests):
        #n_wave = np.random.randint(500, 6500)
        #w0, w1 = np.log10(wgrid[[0, -1]])
        #w = np.logspace(w0, w1, n_wave)
        z = z_array[i]
        r0, r1 = np.log(rgrid[[0, -1]])
        r = np.random.uniform(r0, r1)
        x = np.random.uniform(xgrid[0], xgrid[-1])
        r = np.exp(r)
        w = wgrid

        #test_name = f'{n_wave:04d}_{z:4.3f}_{r:4.3f}_{x:4.3f}'
        test_name = f'z={z:4.3f} R={r:4.3f} xHI={x:4.3f}'
        print(test_name)
        start_time = time.time()
        benchmark = tau_IGM(
            w*(1+z), z, R_ion=r, x_HI_global=x)
        elapsed = (time.time()-start_time)*1e3
        print(f'Elapsed {elapsed:7.4f} ms for benchmark')
        start_time = time.time()
        test = tau_igm_inter(w, z, r, x)
        elapsed = (time.time()-start_time)*1e3
        print(f'Elapsed {elapsed:7.4f} ms for test')
        
        rel_diff = (np.exp(-benchmark+test)-1)
        rel_diff = np.where(
            (np.exp(-benchmark)<1.e-8) & (np.exp(-test)<1.e-8),
            0., rel_diff)
        color = cmap(norm(z))
        axes.step(
             #w*1e4, (np.exp(-benchmark+test)-1)*1e6, label=test_name,
             w/1e4, rel_diff*1e6, label=test_name,
             color=color, lw=1.5)
    axes.set_ylim(-24, 24)
    axes.semilogx()
    axes.tick_params(labelsize=20,)
    axes.set_xlabel('$\mathrm{\lambda_{emit.} \; [\\mu m]}$', fontsize=25)
    axes.set_ylabel('$\mathrm{T_{interp.}/T_{JW}-1 \; [ppm]}$', fontsize=25)
    legend = axes.legend(
        frameon=False, ncol=2, fontsize=25, scatterpoints=1, markerscale=2,
        handletextpad=0.3, labelspacing=0, handlelength=1.2,
        loc='upper left')#, loc='lower left', bbox_to_anchor=(0., 1.))
    for line in legend.get_lines(): line.set_lw(3)
    plt.savefig('transmission_uncertainties.png')
