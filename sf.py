from __future__ import print_function, division
import sys
import os
import rhalphalib as rl
import numpy as np
import scipy.stats
import pickle
import uproot


def get_templ(f, sample, syst=None, sumw2=True):
    hist_name = sample
    if syst is not None:
        hist_name += "_" + syst
    h_vals = f[hist_name].values
    h_edges = f[hist_name].edges
    h_variances = f[hist_name].variances
    h_key = 'msd'
    if not sumw2:
        return (h_vals, h_edges, h_key)
    else:
        return (h_vals, h_edges, h_key, h_variances)

year = 16
f = {
    'pass': uproot.open("wfitsjer{}/wfit_nskim{}_n2cvb/wtag_var_pass.root".format(year, year)),
    'fail': uproot.open("wfitsjer{}/wfit_nskim{}_n2cvb/wtag_var_fail.root".format(year, year)),
    'passpass': uproot.open("wfitsjer{}/wfit_nskim{}_cvl/wtag_var_pass.root".format(year, year)),
    'passfail': uproot.open("wfitsjer{}/wfit_nskim{}_cvl/wtag_var_fail.root".format(year, year)),
}


def test_sfmodel(tmpdir):
    sys_scale = rl.NuisanceParameter('CMS_scale', 'shape')
    sys_smear = rl.NuisanceParameter('CMS_smear', 'shape')
    lumi = rl.NuisanceParameter('CMS_lumi', 'lnN')
    jecs = rl.NuisanceParameter('CMS_jecs', 'lnN')
    pu = rl.NuisanceParameter('CMS_pu', 'lnN')
    effSF = rl.IndependentParameter('effSF', 1., -20, 100)
    effSF2 = rl.IndependentParameter('effSF_un', 1., -20, 100)
    effwSF = rl.IndependentParameter('effwSF', 1., -20, 100)
    effwSF2 = rl.IndependentParameter('effwSF_un', 1., -20, 100)

    msdbins = np.linspace(40, 201, 24)
    msd = rl.Observable('msd', msdbins)
    model = rl.Model("sfModel")

    # for region in ['pass', 'fail']:
    for region in ['passpass', 'passfail', 'fail']:
        ch = rl.Channel("wsf{}".format(region))

        isPass = region == 'pass'
        # made up template
        # print(get_templ(f[region], 'catp2', 'scaleUp', False))
        wqq_templ = get_templ(f[region], 'catp2')
        wqq_sample = rl.TemplateSample("{}_wqq".format(ch.name), rl.Sample.SIGNAL, wqq_templ)
        wqq_sample.setParamEffect(sys_scale, 
                                  get_templ(f[region], 'catp2', 'scaleUp', False),
                                  get_templ(f[region], 'catp2', 'scaleDown', False),
                                  scale=1,
                                  )
        wqq_sample.setParamEffect(sys_smear, 
                                  get_templ(f[region], 'catp2', 'smearUp', False),
                                  get_templ(f[region], 'catp2', 'smearDown', False),
                                  scale=0.1,
                                  )
        wqq_sample.setParamEffect(lumi, 1.023)
        wqq_sample.setParamEffect(jecs, 1.02)
        wqq_sample.setParamEffect(pu, 1.05)
        wqq_sample.autoMCStats()
        ch.addSample(wqq_sample)

        qcd_templ = get_templ(f[region], 'catp1')
        qcd_sample = rl.TemplateSample("{}_qcd".format(ch.name), rl.Sample.BACKGROUND, qcd_templ)
        qcd_sample.setParamEffect(lumi, 1.023)
        qcd_sample.setParamEffect(jecs, 1.02)
        qcd_sample.setParamEffect(pu, 1.05)
        qcd_sample.autoMCStats()
        ch.addSample(qcd_sample)

        data_obs = get_templ(f[region], 'data_obs')[:-1]
        ch.setObservation(data_obs)

        model.addChannel(ch)

    pass_sample1 = model['wsfpasspass']['wqq']
    pass_sample2 = model['wsfpassfail']['wqq']
    fail_sample = model['wsffail']['wqq']
    pass_fail = (pass_sample1.getExpectation(nominal=True).sum() + 
                 pass_sample2.getExpectation(nominal=True).sum()) / fail_sample.getExpectation(nominal=True).sum()
    pass_sample1.setParamEffect(effSF, 1.0 * effSF)
    pass_sample2.setParamEffect(effSF, 1.0 * effSF)
    fail_sample.setParamEffect(effSF, (1 - effSF) * pass_fail + 1)


    pass_sample1 = model['wsfpasspass']['qcd']
    pass_sample2 = model['wsfpassfail']['qcd']
    fail_sample = model['wsffail']['qcd']
    pass_fail = ( pass_sample1.getExpectation(nominal=True).sum() + 
                  pass_sample2.getExpectation(nominal=True).sum()) / fail_sample.getExpectation(nominal=True).sum()
    pass_sample1.setParamEffect(effSF2, 1.0 * effSF2)
    pass_sample2.setParamEffect(effSF2, 1.0 * effSF2)
    fail_sample.setParamEffect(effSF2, (1 - effSF2) * pass_fail + 1)

    pass_sample = model['wsfpasspass']['wqq']
    fail_sample = model['wsfpassfail']['wqq']
    pass_fail = pass_sample.getExpectation(nominal=True).sum() / fail_sample.getExpectation(nominal=True).sum()
    pass_sample.setParamEffect(effwSF, 1.0 * effwSF)
    fail_sample.setParamEffect(effwSF, (1 - effwSF) * pass_fail + 1)

    pass_sample = model['wsfpasspass']['qcd']
    fail_sample = model['wsfpassfail']['qcd']
    pass_fail = pass_sample.getExpectation(nominal=True).sum() / fail_sample.getExpectation(nominal=True).sum()
    pass_sample.setParamEffect(effwSF2, 1.0 * effwSF2)
    fail_sample.setParamEffect(effwSF2, (1 - effwSF2) * pass_fail + 1)

    model.renderCombine('dsf')

if __name__ == '__main__':
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    test_sfmodel('tmp')