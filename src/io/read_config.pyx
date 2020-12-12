# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import ConfigParser
import os, sys

cpdef read_param(param_type, param_name, dtype, usr_dir):

  cdef int rank=0
  IF MPI: rank=mpi.COMM_WORLD.Get_rank()

  param=0.

  if rank==0:

    usr_cfg_path = os.path.join(usr_dir,'config.cfg')

    if os.path.exists(usr_cfg_path):

      config = ConfigParser.ConfigParser()
      config.read(usr_cfg_path)
      if dtype=='f':
        param = config.getfloat(param_type, param_name)
      elif dtype=='i':
        param = config.getint(param_type, param_name)
      elif dtype=='s':
        param = config.get(param_type, param_name)

    else:
      print('error: cannot read user parameter {}:'+
        'unable to open the input file {}\n'.format(param_name, usr_cfg_path))

  IF MPI: param = mpi.COMM_WORLD.bcast(param, root=0)

  return param
