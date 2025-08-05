/**
 * Types for C API.
 */
/** A pointer to points to the raw address space. */
export type Pointer = number;
/** A pointer offset, need to add a base address to get a valid ptr. */
export type PtrOffset = number;
/**
 * Size of common data types.
 */
export declare const enum SizeOf {
    U8 = 1,
    U16 = 2,
    I32 = 4,
    I64 = 8,
    F32 = 4,
    F64 = 8,
    TVMValue = 8,
    TVMFFIAny = 16,
    DLDataType = 4,
    DLDevice = 8,
    ObjectHeader = 16
}
/**
 * Type Index in new TVM FFI.
 *
 * We are keeping the same style as C API here.
 */
export declare const enum TypeIndex {
    kTVMFFINone = 0,
    /*! \brief POD int value */
    kTVMFFIInt = 1,
    /*! \brief POD bool value */
    kTVMFFIBool = 2,
    /*! \brief POD float value */
    kTVMFFIFloat = 3,
    /*! \brief Opaque pointer object */
    kTVMFFIOpaquePtr = 4,
    /*! \brief DLDataType */
    kTVMFFIDataType = 5,
    /*! \brief DLDevice */
    kTVMFFIDevice = 6,
    /*! \brief DLTensor* */
    kTVMFFIDLTensorPtr = 7,
    /*! \brief const char**/
    kTVMFFIRawStr = 8,
    /*! \brief TVMFFIByteArray* */
    kTVMFFIByteArrayPtr = 9,
    /*! \brief R-value reference to ObjectRef */
    kTVMFFIObjectRValueRef = 10,
    /*! \brief Start of statically defined objects. */
    kTVMFFIStaticObjectBegin = 64,
    /*!
     * \brief Object, all objects starts with TVMFFIObject as its header.
     * \note We will also add other fields
     */
    kTVMFFIObject = 64,
    /*!
     * \brief String object, layout = { TVMFFIObject, TVMFFIByteArray, ... }
     */
    kTVMFFIStr = 65,
    /*!
     * \brief Bytes object, layout = { TVMFFIObject, TVMFFIByteArray, ... }
     */
    kTVMFFIBytes = 66,
    /*! \brief Error object. */
    kTVMFFIError = 67,
    /*! \brief Function object. */
    kTVMFFIFunction = 68,
    /*! \brief Array object. */
    kTVMFFIArray = 69,
    /*! \brief Map object. */
    kTVMFFIMap = 70,
    /*!
     * \brief Shape object, layout = { TVMFFIObject, { const int64_t*, size_t }, ... }
     */
    kTVMFFIShape = 71,
    /*!
     * \brief NDArray object, layout = { TVMFFIObject, DLTensor, ... }
     */
    kTVMFFINDArray = 72,
    /*! \brief Runtime module object. */
    kTVMFFIModule = 73
}
/** void* TVMWasmAllocSpace(int size); */
export type FTVMWasmAllocSpace = (size: number) => Pointer;
/** void TVMWasmFreeSpace(void* data); */
export type FTVMWasmFreeSpace = (ptr: Pointer) => void;
/** const char* TVMFFIWasmGetLastError(); */
export type FTVMFFIWasmGetLastError = () => Pointer;
/**
 * int TVMFFIWasmSafeCallType(void* self, const TVMFFIAny* args,
 *                            int32_t num_args, TVMFFIAny* result);
 */
export type FTVMFFIWasmSafeCallType = (self: Pointer, args: Pointer, num_args: number, result: Pointer) => number;
/**
 * int TVMFFIWasmFunctionCreate(void* resource_handle, TVMFunctionHandle* out);
 */
export type FTVMFFIWasmFunctionCreate = (resource_handle: Pointer, out: Pointer) => number;
/**
 * void TVMFFIWasmFunctionDeleter(void* self);
 */
export type FTVMFFIWasmFunctionDeleter = (self: Pointer) => void;
/**
 * int TVMFFIObjectFree(TVMFFIObjectHandle obj);
 */
export type FTVMFFIObjectFree = (obj: Pointer) => number;
/**
 * int TVMFFITypeKeyToIndex(const TVMFFIByteArray* type_key, int32_t* out_tindex);
 */
export type FTVMFFITypeKeyToIndex = (type_key: Pointer, out_tindex: Pointer) => number;
/**
 * int TVMFFIAnyViewToOwnedAny(const TVMFFIAny* any_view, TVMFFIAny* out);
 */
export type FTVMFFIAnyViewToOwnedAny = (any_view: Pointer, out: Pointer) => number;
/**
 * void TVMFFIErrorSetRaisedFromCStr(const char* kind, const char* message);
 */
export type FTVMFFIErrorSetRaisedFromCStr = (kind: Pointer, message: Pointer) => void;
/**
 * int TVMFFIFunctionSetGlobal(const TVMFFIByteArray* name, TVMFFIObjectHandle f,
 *                             int override);
 */
export type FTVMFFIFunctionSetGlobal = (name: Pointer, f: Pointer, override: number) => number;
/**
 * int TVMFFIFunctionGetGlobal(const TVMFFIByteArray* name, TVMFFIObjectHandle* out);
 */
export type FTVMFFIFunctionGetGlobal = (name: Pointer, out: Pointer) => number;
/**
 * int TVMFFIFunctionCall(TVMFFIObjectHandle func, TVMFFIAny* args, int32_t num_args,
 *                        TVMFFIAny* result);
 */
export type FTVMFFIFunctionCall = (func: Pointer, args: Pointer, num_args: number, result: Pointer) => number;
/**
 * int TVMFFIDataTypeFromString(const TVMFFIByteArray* str, DLDataType* out);
 */
export type FTVMFFIDataTypeFromString = (str: Pointer, out: Pointer) => number;
/**
 * int TVMFFIDataTypeToString(const DLDataType* dtype, TVMFFIObjectHandle* out);
 */
export type FTVMFFIDataTypeToString = (dtype: Pointer, out: Pointer) => number;
/**
 * TVMFFITypeInfo* TVMFFIGetTypeInfo(int32_t type_index);
 */
export type FTVMFFIGetTypeInfo = (type_index: number) => Pointer;
