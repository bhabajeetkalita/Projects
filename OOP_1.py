#Author: Bhabajeet Kalita
#Date: 20/09/2017
#Description: An example to show the working of Method Overloading
class GetSetPartent(object):
    __metaclass__=abc.ABCMeta

    def __init__(self,value):
        self.value = 0
    def set_val(self,value):
        self.val = value
    def get_val(self):
        return self.val

    @abc.abstractmethod
    def showdoc(self):
        return

class GetSetList(GetSetPartent):
    def __init__(self,value=0):
        self.vallist=[ value ]
    def get_val(self):
        return self.vallist[-1]
    def get_vals(self):
        return self.vallist
    def set_val(self,value):
        self.vallist.append(value)
    def showdoc(self):
        print("GetSetList object',len{0},stores history of values".format(len(self.vallist)))
gsl=GetSetList(5)
gsl.set_val(10)
