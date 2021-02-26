#!/usr/bin/env python
# coding: utf-8

# In[11]:


from emc.workflows.correct import init_emc_wf
from pathlib import Path
import nipype.pipeline.engine as pe
from nipype.interfaces import utility as niu
from dipy.data import get_fnames
from dmriprep.workflows.dwi.util import init_dwi_reference_wf
from dmriprep.interfaces.vectors import CheckGradientTable

#fdata, fbvals, fbvecs = get_fnames("sherbrooke_3shell")
fdata, fbvals, fbvecs = get_fnames("stanford_hardi")


# In[12]:


# Create a workflow that can be added as a node to
# `~dmriprep.workflows.dwi.base.init_dwi_preproc_wf` and (for now) envelops
# `~dmriprep.workflows.dwi.util.init_dwi_reference_wf`
def build_emc_workflow(fdata, fbvals, fbvecs, work_dir='/tmp'):

    wf_emc = pe.Workflow(name='emc_correction')

    inputnode = pe.Node(niu.IdentityInterface(fields=['dwi_files', 'in_bval',
                                                      'in_bvec']),
        name='inputnode')

    inputnode.inputs.dwi_file = fdata
    inputnode.inputs.in_bval = fbvals
    inputnode.inputs.in_bvec = fbvecs

    emc_wf_node = init_emc_wf('emc_wf')

    wf_emc.connect([
        (inputnode, emc_wf_node, [('dwi_file', 'meta_inputnode.dwi_file'),
                                  ('in_bval', 'meta_inputnode.in_bval'),
                                  ('in_bvec', 'meta_inputnode.in_bvec')])
    ])

    wf_emc.base_dir = work_dir

    cfg = dict(execution={'stop_on_first_crash': False,
                          'parameterize_dirs': True, 'crashfile_format': 'txt',
                          'remove_unnecessary_outputs': False,
                          'poll_sleep_duration': 0, 'plugin': 'MultiProc'})
    for key in cfg.keys():
        for setting, value in cfg[key].items():
            wf_emc.config[key][setting] = value

    return wf_emc


# In[13]:


wf = build_emc_workflow(fdata, fbvals, fbvecs)
res = wf.run(plugin='MultiProc')


# In[ ]:


list(res.nodes())


# In[30]:





# In[ ]:




