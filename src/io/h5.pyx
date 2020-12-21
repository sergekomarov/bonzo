# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py.libmpi cimport MPI_COMM_WORLD, MPI_INFO_NULL
  from mpi4py.libmpi cimport MPI_Comm_size, MPI_Comm_rank
  from mpi4py.libmpi cimport MPI_Init, MPI_Finalize

from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np

IF SPREC:
  H5_REAL = H5T_NATIVE_FLOAT
ELSE:
  H5_REAL = H5T_NATIVE_DOUBLE


cdef void h5write(char *file_name, object dsets, object attrs, int ndim,
                  int *grid_dim, int *chunk_dim, int *block_dim,
                  int *offset, int chunked):

  cdef:
    hid_t file_id, dset_id, attr_id          # identifiers
    hid_t filespace, memspace, attrspace     # file and memory dataspace identifiers
    hsize_t	count[3]	                       # hyperslab selection parameters
    hsize_t	stride[3]
    hid_t	plist_id   # property list identifier

    int n
    herr_t	status
    void *data_buff
    hid_t data_type

  IF MPI:

    cdef:
      int mpi_size, mpi_rank
      MPI_Comm comm  = MPI_COMM_WORLD
      MPI_Info info  = MPI_INFO_NULL

    MPI_Comm_size(comm, &mpi_size)
    MPI_Comm_rank(comm, &mpi_rank)

  # Set up file access property list with parallel I/O access.
  plist_id = H5Pcreate(H5P_FILE_ACCESS)
  IF MPI: H5Pset_fapl_mpio(plist_id, comm, info)

  # Create a new file collectively and release property list identifier.
  file_id = H5Fcreate(<char*>file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id)
  H5Pclose(plist_id)

  # Parameters to locate the chunk on the grid.
  for n in range(ndim):
    count[n]  = 1
    stride[n] = 1

  # Iterate over datasets and write them.

  for (dset_name, dset_data) in dsets.items():

    if ndim==3:
      data_buff = alloc_buffer3(dset_data[1], block_dim):
      write_into_buffer3(data_buff, dset_data, block_dim)
    else:
      data_buff = &((dset_data[0])[0])

    data_type = translate_dtype(dset_data[1])

    filespace = H5Screate_simple(ndim, <hsize_t*>grid_dim,  NULL)
    memspace  = H5Screate_simple(ndim, <hsize_t*>block_dim, NULL)

    # Create chunked dataset and close filespace.
    plist_id = H5Pcreate(H5P_DATASET_CREATE)
    if chunked:
      H5Pset_chunk(plist_id, ndim, <hsize_t*>chunk_dim)
    dset_id = H5Dcreate(file_id, <char*>dset_name, data_type, filespace,
      H5P_DEFAULT, plist_id, H5P_DEFAULT)
    H5Pclose(plist_id)
    H5Sclose(filespace)

    # Select hyperslab in the file.
    filespace = H5Dget_space(dset_id)
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                        <hsize_t*>offset, stride, count, <hsize_t*>block_dim)

    # Create property list for collective dataset write.
    plist_id = H5Pcreate(H5P_DATASET_XFER)
    IF MPI: H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE)
    status = H5Dwrite(dset_id, data_type, memspace, filespace,
          plist_id, data_buff)

    if ndim>1: free(data_buff)

    # Close/release resources.
    H5Dclose(dset_id)
    H5Sclose(filespace)
    H5Sclose(memspace)
    H5Pclose(plist_id)


  # Write attributes.

  for (attr_name, attr_data) in attrs.items():

    data_buff=&(attr_data[0])
    data_type=translate_dtype(attr_data[1])

    # Create scalar attribute.
    attrspace = H5Screate(H5S_SCALAR)
    attr_id   = H5Acreate(file_id, <char*>attr_name, data_type,
                      attrspace, H5P_DEFAULT, H5P_DEFAULT)

    # Write scalar attribute.
    status = H5Awrite(attr_id, data_type, data_buff)

    # Close attribute dataspace.
    status = H5Sclose(attrspace)

    # Close the attributes.
    status = H5Aclose(attr_id)

  # ----------------------------------------------------------------

  H5Fclose(file_id)


# =========================================================================

cdef void h5restart(char *file_name, object dsets, object attrs, int ndim,
    int *block_dim, int *offset):

  cdef:
    hid_t file_id, dset_id, attr_id          # identifiers
    hid_t filespace, memspace, attrspace     # file and memory dataspace identifiers
    hsize_t	count[3]	              # hyperslab selection parameters
    hsize_t	stride[3]
    hid_t	plist_id   # property list identifier

    int n
    herr_t	status
    void *data_buff
    hid_t data_type

  IF MPI:
    cdef:
      int mpi_size, mpi_rank
      MPI_Comm comm  = MPI_COMM_WORLD
      MPI_Info info  = MPI_INFO_NULL

    MPI_Comm_size(comm, &mpi_size)
    MPI_Comm_rank(comm, &mpi_rank)

  # Set up file access property list with parallel I/O access.
  plist_id = H5Pcreate(H5P_FILE_ACCESS)
  IF MPI: H5Pset_fapl_mpio(plist_id, comm, info)

  # Create a new file collectively and release property list identifier.
  file_id = H5Fopen(<char*>file_name, H5F_ACC_RDWR, plist_id)
  H5Pclose(plist_id)

  # Parameters to locate the chunk on the grid.
  for n in range(ndim):
    count[n]  = 1
    stride[n] = 1

  # Iterate over datasets and write them.

  for (dset_name, dset_data) in dsets.items():

    # Open dataset.
    dset_id = H5Dopen(file_id, <char*>dset_name, H5P_DEFAULT)

    # Select hyperslab in the file.
    filespace = H5Dget_space(dset_id)
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                        <hsize_t*>offset, stride, count, <hsize_t*>block_dim)

    memspace  = H5Screate_simple(ndim, <hsize_t*>block_dim, NULL)

    if ndim==3: data_buff =alloc_buffer3(dset_data[1], block_dim)
    else: data_buff = &((dset_data[0])[0])

    data_type=translate_dtype(dset_data[1])

    # Create property list for collective dataset read.
    plist_id = H5Pcreate(H5P_DATASET_XFER)
    IF MPI: H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE)

    # Read data into buffer.
    status = H5Dread(dset_id, data_type, memspace, filespace,
                plist_id, data_buff)

    if ndim==3:
      read_from_buffer3(dset_data, data_buff, block_dim)
      free(data_buff)

    # Close/release resources.
    H5Pclose(plist_id)
    H5Sclose(memspace)
    H5Sclose(filespace)
    H5Dclose(dset_id)


  # Read attributes.

  for (attr_name, attr_data) in attrs.items():

    data_buff = &(attr_data[0])
    data_type = translate_dtype(attr_data[1])

    # Create scalar attribute.
    attr_id = H5Aopen(file_id, <char*>attr_name, H5P_DEFAULT)
    attrspace = H5Aget_space(attr_id)

    # Read scalar attribute.
    H5Aread(attr_id, data_type, data_buff)

    # Close attribute dataspace.
    status = H5Sclose(attrspace)

    # Close the attributes.
    status = H5Aclose(attr_id)

  # ----------------------------------------------------------------

  H5Fclose(file_id)


# ==============================================================================

cdef void write_into_buffer3(void *buff, object dset_data, int *block_dim):

  for k in range(block_dim[0]):
    for j in range(block_dim[1]):
      for i in range(block_dim[2]):

        if dset_data[1]=='r':
          (<real*>buff)[k*block_dim[1]*block_dim[2] + j*block_dim[2] + i] = dset_data[0][k,j,i]
        elif dset_data[1]=='i':
          (<int*>buff)[ k*block_dim[1]*block_dim[2] + j*block_dim[2] + i] = dset_data[0][k,j,i]
        if dset_data[1]=='l':
          (<long*>buff)[k*block_dim[1]*block_dim[2] + j*block_dim[2] + i] = dset_data[0][k,j,i]
        if dset_data[1]=='s':
          (<char*>buff)[k*block_dim[1]*block_dim[2] + j*block_dim[2] + i] = dset_data[0][k,j,i]


cdef void read_from_buffer3(object dset_data, void *buff, int *block_dim):

  cdef:
    int i,j,k
    long n

  for k in range(block_dim[0]):
    for j in range(block_dim[1]):
      for i in range(block_dim[2]):

        n = k*block_dim[1]*block_dim[2] + j*block_dim[2] + i

        if dset_data[1]=='r':
          dset_data[0][k,j,i] = (<real*>buff)[n]
        elif dset_data[1]=='i':
          dset_data[0][k,j,i] = (<int*>buff)[n]
        elif dset_data[1]=='l':
          dset_data[0][k,j,i] = (<long*>buff)[n]
        elif dset_data[1]=='s':
          dset_data[0][k,j,i] = (<char*>buff)[n]


cdef void* alloc_buffer3(data_type, int *block_dim):

  cdef long data_len = block_dim[0]*block_dim[1]*block_dim[2]

  if data_type=='r':
    return malloc(data_len*sizeof(real))
  elif data_type=='i':
    return malloc(data_len*sizeof(int))
  elif data_type=='l':
    return malloc(data_len*sizeof(long))
  elif data_type=='s':
    return malloc(data_len*sizeof(char))
  else:
    return NULL


cdef hid_t translate_dtype(a):
  if a=='i':
    return H5T_NATIVE_INT
  elif a=='l':
    return H5T_NATIVE_LONG
  elif a=='r':
    return H5_REAL
  elif a=='s':
    return H5T_NATIVE_CHAR
  else:
    return H5_REAL
