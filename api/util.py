"""
API 관련 util 함수들의 모임~

"""
# Author: Haneul Kim <haneulkim214@gmail.com>
# License: Enliple


from error_handler import Errors, InputError


def check_json(inp_data):
    """
    Check datatype of each value in json object and raise
    InputError if wrong datatype.

    Parameters
    ----------
    inp_data : JSON
        Independent variables used for prediction.
    """
    dtype_dict = {"createdDate":str, "mediaId":str, "inventoryId":int,
                "adverId":str, "platform":str, "cpoint":float, "mpoint":float,
                "freqLog":int, "tTime":int}

    for col_nm, dtype_ in dtype_dict.items():
        if type(inp_data[col_nm]) != dtype_:
            raise InputError(f"{col_nm} must be in {dtype_} datatype")