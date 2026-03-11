from dataclasses import asdict

class ObjectDataDictionary:
    """
        A class containing data structure and methods to enable clean 
        data storage
    """
    def __init__(self):
        self.data = {} 
        # All data will be appended to this dictionary as and when 
        # the append defs are cllaed after updateoing every data field,
        # such as material or ID data field for example.

    def append(self, DataClassObject):
        self.data[DataClassObject.__class__.__name__] = asdict(DataClassObject)
        
    def CrossCheckAndAppend(self, DataClassObject):
        if DataClassObject.__class__.__name__ not in self.data.keys():
            ObjectDataDictionary.append(DataClassObject)
        elif DataClassObject.__class__.__name__ in self.data.keys():
            D = asdict(DataClassObject)
            for key in self.data[DataClassObject.__class__.__name__].keys():
                print(key)
                if self.data[DataClassObject.__class__.__name__][key] != D[key]:
                    self.data[DataClassObject.__class__.__name__][key] = D[key]
                else:
                    pass