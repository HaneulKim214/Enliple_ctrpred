from const import model2use


class Errors(Exception):
    def __init__(self, message):
        super(Errors, self).__init__(message)
        self.message = message

    def response(self):
        return {"expected_click":"",
                "click_%":"",
                "model":model2use,
                "status":self.status_code,
                "message":self.message
                }


class ModelError(Errors):
    """For handling Errors associated with Models"""
    def __init__(self, message, status_code="01"):
        self.message = message
        self.status_code = status_code


class InputError(Errors):
    def __init__(self, message, status_code="02"):
        self.message = message
        self.status_code = status_code


class ValueError(Errors):
    def __init__(self, message, status_code="03"):
        self.message = message
        self.status_code = status_code