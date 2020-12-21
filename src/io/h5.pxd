# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py.libmpi cimport MPI_Comm, MPI_Info
  from bnz.defs cimport *

cdef extern from "hdf5.h":

  ctypedef long int hid_t
  ctypedef int hbool_t
  ctypedef int herr_t
  ctypedef int htri_t
  ctypedef long long hsize_t
  ctypedef signed long long hssize_t
  ctypedef signed long long haddr_t
  ctypedef long int off_t

# === H5F - File API ==========================================================

  # File constants
  cdef enum:
    H5F_ACC_TRUNC
    H5F_ACC_RDONLY
    H5F_ACC_RDWR
    H5F_ACC_EXCL
    H5F_ACC_DEBUG
    H5F_ACC_CREAT
    H5F_ACC_SWMR_WRITE
    H5F_ACC_SWMR_READ

  # HDF5 uses a clever scheme wherein these are actually init() calls
  # Hopefully Cython won't have a problem with this.
  # Thankfully they are defined but -1 if unavailable
  hid_t H5FD_CORE
  hid_t H5FD_FAMILY
  hid_t H5FD_LOG
  hid_t H5FD_MPIO
  hid_t H5FD_MULTI
  hid_t H5FD_SEC2
  hid_t H5FD_STDIO

  ctypedef enum H5FD_mpio_xfer_t:
    H5FD_MPIO_INDEPENDENT = 0,
    H5FD_MPIO_COLLECTIVE


# === H5P - Property list API =================================================

  int H5P_DEFAULT

  # Property list classes
  hid_t H5P_NO_CLASS
  hid_t H5P_FILE_CREATE
  hid_t H5P_FILE_ACCESS
  hid_t H5P_DATASET_CREATE
  hid_t H5P_DATASET_ACCESS
  hid_t H5P_DATASET_XFER


# === H5S - Dataspaces ========================================================

  # Codes for defining selections
  ctypedef enum H5S_seloper_t:
    H5S_SELECT_NOOP      = -1,
    H5S_SELECT_SET       = 0,
    H5S_SELECT_OR,
    H5S_SELECT_AND,
    H5S_SELECT_XOR,
    H5S_SELECT_NOTB,
    H5S_SELECT_NOTA,
    H5S_SELECT_APPEND,
    H5S_SELECT_PREPEND,
    H5S_SELECT_INVALID    # Must be the last one

  ctypedef enum H5S_class_t:
    H5S_NO_CLASS         = -1,  #/*error
    H5S_SCALAR           = 0,   #/*scalar variable
    H5S_SIMPLE           = 1,   #/*simple data space
    H5S_NULL             = 2,   # NULL data space
    # no longer defined in 1.8
    #H5S_COMPLEX          = 2    #/*complex data space


# === H5T - Datatypes =========================================================

  # --- Predefined datatypes --------------------------------------------------

  cdef hid_t H5T_NATIVE_B8
  cdef hid_t H5T_NATIVE_CHAR
  cdef hid_t H5T_NATIVE_SCHAR
  cdef hid_t H5T_NATIVE_UCHAR
  cdef hid_t H5T_NATIVE_SHORT
  cdef hid_t H5T_NATIVE_USHORT
  cdef hid_t H5T_NATIVE_INT
  cdef hid_t H5T_NATIVE_UINT
  cdef hid_t H5T_NATIVE_LONG
  cdef hid_t H5T_NATIVE_ULONG
  cdef hid_t H5T_NATIVE_LLONG
  cdef hid_t H5T_NATIVE_ULLONG
  cdef hid_t H5T_NATIVE_FLOAT
  cdef hid_t H5T_NATIVE_DOUBLE
  cdef hid_t H5T_NATIVE_LDOUBLE

  cdef hid_t H5T_NATIVE_INT8
  cdef hid_t H5T_NATIVE_UINT8
  cdef hid_t H5T_NATIVE_INT16
  cdef hid_t H5T_NATIVE_UINT16
  cdef hid_t H5T_NATIVE_INT32
  cdef hid_t H5T_NATIVE_UINT32
  cdef hid_t H5T_NATIVE_INT64
  cdef hid_t H5T_NATIVE_UINT64

  # Functions.

  hid_t  H5Pcreate(hid_t plist_id)
  hid_t  H5Fopen(char *name, unsigned int flags, hid_t fapl_id)
  herr_t H5Pset_chunk(hid_t plist, int ndims, hsize_t * dim)
  herr_t H5Pset_fapl_mpio(hid_t fapl_id, MPI_Comm comm, MPI_Info info)
  herr_t H5Pset_dxpl_mpio( hid_t dxpl_id, H5FD_mpio_xfer_t xfer_mode )
  herr_t H5Pclose(hid_t plist_id)

  hid_t  H5Fcreate(char *filename, unsigned int flags, hid_t create_plist, hid_t access_plist)
  herr_t H5Fclose(hid_t file_id)

  hid_t  H5Screate(H5S_class_t type)
  hid_t  H5Screate_simple(int rank, hsize_t *dims, hsize_t *maxdims)
  herr_t H5Sselect_hyperslab(hid_t space_id, H5S_seloper_t op,  hsize_t *start, hsize_t *_stride, hsize_t *count, hsize_t *_block)
  herr_t H5Sclose(hid_t space_id)

  hid_t  H5Dcreate(hid_t loc_id, char *name, hid_t type_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id) nogil
  hid_t  H5Dopen(hid_t loc_id, char *name, hid_t dapl_id)
  hid_t  H5Dget_space(hid_t dset_id)
  herr_t H5Dwrite(hid_t dset_id, hid_t mem_type, hid_t mem_space, hid_t file_space, hid_t xfer_plist, void* buf) nogil
  herr_t H5Dread(hid_t dataset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t xfer_plist_id, void *buf)
  herr_t H5Dclose(hid_t dset_id)

  hid_t  H5Acreate(hid_t loc_id, char *name, hid_t type_id, hid_t space_id, hid_t acpl_id, hid_t aapl_id)
  hid_t  H5Aopen(hid_t obj_id, char *attr_name, hid_t aapl_id)
  herr_t H5Awrite(hid_t attr_id, hid_t mem_type_id, void *buf)
  hid_t  H5Aget_space(hid_t attr_id)
  herr_t H5Aread(hid_t attr_id, hid_t mem_type_id, void *buf)
  herr_t H5Aclose(hid_t attr_id)


# ==============================================================================
cdef void h5write(  char*, object, object, int, int*, int*, int*, int*, int)
cdef void h5restart(char*, object, object, int, int*, int*)
