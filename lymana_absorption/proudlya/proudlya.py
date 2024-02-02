#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""proudly.py - classes and methods for storing parameters and predicting
observed spectra and photometry from them, given a Source object.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sedpy.observate import getSED

from prospect.sources.constants import lightspeed, jansky_cgs
from prospect.models.sedmodel import SpecModel, PolySpecModel

from lymana_absorption.mean_IGM_absorption import igm_absorption
from lymana_absorption.lymana_optical_depth import tau_DLA, tau_IGM

__all__ = [
           "DLASpecModel", "DLAPolySpecModel",
]


class DLASpecModel(SpecModel):
    """Class to fit simultaneously a SFH from the `SpecModel` class,
    Damped Lyman Alpha (DLA) absorption and IGM absorption.

    In addition to the parameters from `prospect.models.SpecModel`, the
    following parameters are available.
        DLA_logN_HI     : column density toward the DLA
        DLA_T_HI        : temperature of the HI gas
        DLA_b_turb      : turbulence parameter of the HI gas
        IGM_R_ion       : radius of the ionised bubble around the galaxy
        IGM_x_HI_global : global neutral fraction
        IGM_cosmo       : instance of `astropy.cosmology.cosmology`.

    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def _available_parameters(self):
        pars = [("DLA_logN_HI",
                 "density of HI in the DLA, [cm^-2]"),
                ("DLA_T_HI", "Temperature of the HI [K]"),
                ("DLA_b_turb", "Turbulence parameter of the gas"),
                ("IGM_interp", "Callable to calculate the IGM opacity"),
                ("IGM_R_ion", "Radium of ionised bubble around source [Mpc]"),
                ("IGM_x_HI_global", "Global neutral fraction [0-1]. Not bubble x_HI!"),
                ("IGM_cosmo", "Cosmology used for IGM calculations [Need checking]")
                ]

        return pars

    def __cache_DLA_transmission__(self):

        assert getattr(self, '_wave', None) is not None, (
            'Call `predict` first, to assign a model wavelength grid')

        logN_HI = self.params.get('DLA_logN_HI', 0.)
        T_HI    = self.params.get('DLA_T_HI',  1.e4)
        b_turb  = self.params.get('DLA_b_turb',  0.)

        self._zred = self.params.get('zred', 0)

        tau = tau_DLA(
            wl_emit_array=self._wave, N_HI=10.**logN_HI,
            T=T_HI, b_turb=b_turb)

        self.dla_transmission = np.exp(-tau)


    def __cache_IGM_transmission__(self):

        assert getattr(self, '_wave', None) is not None, (
            'Call `predict` first, to assign a model wavelength grid')

        R_ion       = self.params.get('IGM_R_ion',   np.nan)
        x_HI_global = self.params.get('IGM_x_HI_global', np.nan)
        igm_cosmo   = self.params.get('IGM_cosmo',   None)

        assert np.isnan(R_ion)==np.isnan(x_HI_global)==(igm_cosmo is None), (
            '"IGM_R_ion", "x_HI_global" and "IGM_cosmo" must be used together')
       
        self._zred = self.params.get('zred', 0)

        # This is done and double checked that wavelength is right.
        self.img_transmission = igm_absorption(
             self._wave*(1+self._zred), self._zred)

        # For simple models without IGM.
        if igm_cosmo is None:
            self.igm_transmission *= np.where(self._wave<1215.6, 0., 1.)
            return 

        tau = tau_IGM(
            wl_obs_array=self._wave*(1+self._zred), z_s=self._zred,
            R_ion=R_ion, x_HI_global=x_HI_global, cosmo=igm_cosmo)
            
        self.igm_transmission *= np.exp(-tau)

        

    def predict(self, theta, observations=None, sps=None, **extras):
        """Given a ``theta`` vector, generate a spectrum, photometry, and any
        extras (e.g. stellar mass), including any calibration effects.

        Parameters
        ----------
        theta : ndarray of shape ``(ndim,)``
            Vector of free model parameter values.

        observations : A list of `Observation` instances (e.g. instance of )
            The data to predict

        sps :
            An `sps` object to be used in the model generation.  It must have
            the :py:func:`get_galaxy_spectrum` method defined.

        Returns
        -------
        predictions: (list of ndarrays)
            List of predictions for the given list of observations.

            If the observation kind is "spectrum" then this is the model spectrum for these
            parameters, at the wavelengths specified by ``obs['wavelength']``,
            including multiplication by the calibration vector.  Units of
            maggies

            If the observation kind is "photometry" then this is the model
            photometry for these parameters, for the filters specified in
            ``obs['filters']``.  Units of maggies.

        extras :
            Any extra aspects of the model that are returned.  Typically this
            will be `mfrac` the ratio of the surviving stellar mass to the
            stellar mass formed.
        """

        # generate and cache intrinsic model spectrum and info
        self.set_parameters(theta)
        self._wave, self._spec, self._mfrac = sps.get_galaxy_spectrum(**self.params)
        self._zred = self.params.get('zred', 0)
        self._eline_wave, self._eline_lum = sps.get_galaxy_elines()
        self._library_resolution = getattr(sps, "spectral_resolution", 0.0) # restframe

        # Flux normalize
        self._norm_spec = self._spec * self.flux_norm()
        # cache eline parameters
        eline_z = self.params.get("eline_delta_zred", 0.0)
        self._ewave_obs = (1 + eline_z + self._zred) * self._eline_wave
        self._ln_eline_penalty = 0
        # physical velocity smoothing of the whole UV/NIR spectrum
        self._smooth_spec = self.velocity_smoothing(self._wave, self._norm_spec)

        self.__cache_DLA_transmission__()
        self.__cache_IGM_transmission__()

        # generate predictions for likelihood
        # this assumes all spectral datasets (if present) occur first
        # because they can change the line strengths during marginalization.
        predictions = [self.predict_obs(obs) for obs in observations]

        return predictions, self._mfrac



    def predict_spec(self, obs, **extras):
        """Generate a prediction for the observed spectrum.  This method assumes
        that the parameters have been set and that the following attributes are
        present and correct

          + ``_wave`` - The SPS restframe wavelength array
          + ``_zred`` - Redshift
          + ``_norm_spec`` - Observed frame spectral fluxes, in units of maggies
          + ``_eline_wave`` and ``_eline_lum`` - emission line parameters from the SPS model

        It generates the following attributes

          + ``_outwave`` - Wavelength grid (observed frame)
          + ``_speccal`` - Calibration vector
          + ``_sed`` - Intrinsic spectrum (before cilbration vector applied but including emission lines)

        And the following attributes are generated if nebular lines are added

          + ``_fix_eline_spec`` - emission line spectrum for fixed lines, intrinsic units
          + ``_fix_eline_spec`` - emission line spectrum for fitted lines, with
            spectroscopic calibration factor included.

        Numerous quantities related to the emission lines are also cached (see
        ``cache_eline_parameters()`` and ``fit_mle_elines()`` for details.)

        :param obs:
            An instance of `Spectrum`, containing the output wavelength array,
            the observed fluxes and uncertainties thereon.  Assumed to be the
            result of :py:meth:`utils.obsutils.rectify_obs`

        :param sigma_spec: (optional)
            The covariance matrix for the spectral noise. It is only used for
            emission line marginalization.

        :returns spec:
            The prediction for the observed frame spectral flux these
            parameters, at the wavelengths specified by ``obs['wavelength']``,
            including multiplication by the calibration vector.
            ndarray of shape ``(nwave,)`` in units of maggies.
        """
        # redshift model wavelength
        obs_wave = self.observed_wave(self._wave, do_wavecal=False)

        # get output wavelength vector
        self._outwave = obs.wavelength
        if self._outwave is None:
            self._outwave = obs_wave

        # Set up for emission lines.
        _cache_resolution_ = obs.resolution
        obs.resolution = None # Store and remove resolution of observation; instrumental
                              # resolution must be done after DLA.
                              # Remember to RESTORE this value at line `atohydatdnya`
        self.cache_eline_parameters(obs)

        no_inst_spec = self._smooth_spec

        # --- add fixed lines if necessary ---
        emask = self._fix_eline_pixelmask
        if emask.any():
            inds = self._fix_eline & self._valid_eline
            espec = self.predict_eline_spec(line_indices=inds,
                                            wave=self._outwave[emask])
            self._fix_eline_spec = espec
            no_inst_spec[emask] += self._fix_eline_spec.sum(axis=1)

        # This is line `atohydatdnya`; restoring `obs.resolution' to its value.
        # Frankly, this programming style is an evil version of monkey patching
        # and I disavow it as a programming crime.
        obs.resolution = _cache_resolution_

        # Now add DLA.
        dla_attenuated = no_inst_spec * self.dla_transmission
        dla_attenuated = dla_attenuated * self.igm_transmission

        # --- smooth and put on output wavelength grid ---
        # Instrumental smoothing (accounting for library resolution)
        # Put onto the spec.wavelength grid.
        inst_spec = obs.instrumental_smoothing(obs_wave, dla_attenuated,
                                               libres=self._library_resolution)

        # --- calibration ---
        self._speccal = self.spec_calibration(obs=obs, spec=inst_spec, **extras)
        calibrated_spec = inst_spec * self._speccal

        # --- fit and add lines if necessary ---
        emask = self._fit_eline_pixelmask
        if emask.any():
            # We need the spectroscopic covariance matrix to do emission line optimization and marginalization
            sigma_spec = None
            # FIXME: do this only if the noise model is non-trivial, and make sure masking is consistent
            #vectors = obs.noise.populate_vectors(obs)
            #sigma_spec = obs.noise.construct_covariance(**vectors)
            self._fit_eline_spec = self.fit_mle_elines(obs, calibrated_spec, sigma_spec)
            calibrated_spec[emask] += self._fit_eline_spec.sum(axis=1)

        # --- cache intrinsic spectrum ---
        self._sed = calibrated_spec / self._speccal

        return calibrated_spec

    def predict_lines(self, obs, **extras):
        """Generate a prediction for the observed nebular line fluxes.  This method assumes
        that the model parameters have been set, that any adjustments to the
        emission line fluxes based on ML fitting have been applied, and that the
        following attributes are present and correct
          + ``_wave`` - The SPS restframe wavelength array
          + ``_zred`` - Redshift
          + ``_eline_wave`` and ``_eline_lum`` - emission line parameters from the SPS model
        It generates the following attributes
          + ``_outwave`` - Wavelength grid (observed frame)
          + ``_speccal`` - Calibration vector
          + ``line_norm`` - the conversion from FSPS line luminosities to the
                            observed line luminosities, including scaling fudge_factor
          + ``_predicted_line_inds`` - the indices of the line that are predicted

        Numerous quantities related to the emission lines are also cached (see
        ``cache_eline_parameters()`` and ``fit_mle_elines()`` for details) including
        ``_predicted_line_inds`` which is the indices of the line that are predicted.
        ``cache_eline_parameters()`` and ``fit_elines()`` for details).


        :param obs:
            A ``data.observation.Lines()`` instance, with the attributes
            + ``"wavelength"`` - the observed frame wavelength of the lines.
            + ``"line_ind"`` - a set of indices identifying the observed lines in
            the fsps line array

        :returns elum:
            The prediction for the observed frame nebular emission line flux these
            parameters, at the wavelengths specified by ``obs['wavelength']``,
            ndarray of shape ``(nwave,)`` in units of erg/s/cm^2.
        """
        obs_wave = self.observed_wave(self._eline_wave, do_wavecal=False)
        self._outwave = obs.get('wavelength', obs_wave)
        assert len(self._outwave) <= len(self.emline_info)

        # --- cache eline parameters ---
        self.cache_eline_parameters(obs)

        # find the indices of the observed emission lines
        #dw = np.abs(self._ewave_obs[:, None] - self._outwave[None, :])
        #self._predicted_line_inds = np.argmin(dw, axis=0)
        self._predicted_line_inds = obs.get("line_ind")
        self._speccal = 1.0

        self.line_norm = self.flux_norm() / (1 + self._zred) * (3631*jansky_cgs)
        self.line_norm *= self.params.get("linespec_scaling", 1.0)
        elums = self._eline_lum[self._predicted_line_inds] * self.line_norm

        return elums

    def predict_phot(self, filterset):
        """Generate a prediction for the observed photometry.  This method assumes
        that the parameters have been set and that the following attributes are
        present and correct:
          + ``_wave`` - The SPS restframe wavelength array
          + ``_zred`` - Redshift
          + ``_norm_spec`` - Observed frame spectral fluxes, in units of maggies.
          + ``_ewave_obs`` and ``_eline_lum`` - emission line parameters from
            the SPS model

        :param filters:
            Instance of :py:class:`sedpy.observate.FilterSet` or list of
            :py:class:`sedpy.observate.Filter` objects. If there is no
            photometry, ``None`` should be supplied.

        :returns phot:
            Observed frame photometry of the model SED through the given filters.
            ndarray of shape ``(len(filters),)``, in units of maggies.
            If ``filters`` is None, this returns 0.0
        """
        if filterset is None:
            return 0.0

        # generate photometry w/o emission lines
        obs_wave = self.observed_wave(self._wave, do_wavecal=False)
        flambda = self._norm_spec * lightspeed / obs_wave**2 * (3631*jansky_cgs)

        dla_attenuated = flambda * self.dla_transmission
        dla_attenuated = dla_attenuated * self.igm_transmission

        phot = np.atleast_1d(getSED(obs_wave, dla_attenuated, filterset, linear_flux=True))

        # generate emission-line photometry
        if (self._want_lines & self._need_lines):
            phot += self.nebline_photometry(filterset) #???????TODO

        return phot



    def nebline_photometry(self, filterset, elams=None, elums=None):
        """Compute the emission line contribution to photometry.  This requires
        several cached attributes:
          + ``_ewave_obs``
          + ``_eline_lum``

        :param filters:
            Instance of :py:class:`sedpy.observate.FilterSet` or list of
            :py:class:`sedpy.observate.Filter` objects

        :param elams: (optional)
            The emission line wavelength in angstroms.  If not supplied uses the
            cached ``_ewave_obs`` attribute.

        :param elums: (optional)
            The emission line flux in erg/s/cm^2.  If not supplied uses  the
            cached ``_eline_lum`` attribute and applies appropriate distance
            dimming and unit conversion.

        :returns nebflux:
            The flux of the emission line through the filters, in units of
            maggies. ndarray of shape ``(len(filters),)``
        """
        if (elams is None) or (elums is None):
            elams = self._ewave_obs[self._use_eline]
            # We have to remove the extra (1+z) since this is flux, not a flux density
            # Also we convert to cgs
            self.line_norm = self.flux_norm() / (1 + self._zred) * (3631*jansky_cgs)
            elums = self._eline_lum[self._use_eline] * self.line_norm

        # TODO is this the right wavelength, or 1+z?
        elums *= np.interp(elams, self._wave, self.dla_transmission)
        elums *= np.interp(elams, self._wave, self.igm_transmission)

        # loop over filters
        flux = np.zeros(len(filterset))
        try:
            # TODO: Since in this case filters are on a grid, there should be a
            # faster way to look up the transmission than the later loop
            flist = filterset.filters
        except(AttributeError):
            flist = filterset
        for i, filt in enumerate(flist):
            # calculate transmission at line wavelengths
            trans = np.interp(elams, filt.wavelength, filt.transmission,
                              left=0., right=0.)
            # include all lines where transmission is non-zero
            idx = (trans > 0)
            if True in idx:
                flux[i] = (trans[idx]*elams[idx]*elums[idx]).sum() / filt.ab_zero_counts

        return flux



class DLAPolySpecModel(PolySpecModel, DLASpecModel):
    """Same as `DLAPolySpecModel`, but includes a calibration polynomial to
    scale the spectrum to the level of the photometry."""
    pass
