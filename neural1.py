import numpy
import scipy.special

#neural network class definition

class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learning_rate):
        #set number of nodes in each layer
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        #link weight
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.lr=learning_rate
        self.activation_function=lambda x:scipy.special.expit(x)  #lambda: unanimous function
        pass

    def train(self,inputs_list,targets_list):    #train the network
        inputs=numpy.array(inputs_list,ndmin=2).T
        targets=numpy.array(targets_list,ndmin=2).T

        #calculate sianals into hidden layer
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        #calculate signals into final output layer
        final_inputs=numpy.dot(self.who,hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs=self.activation_function(final_inputs)

        #error is the target-final_putputs
        output_errors=targets-final_outputs
        #hidden layer error
        hidden_errors=numpy.dot(self.who.T,output_errors)
        #updates the weights for the links between the hidden and output layers
        self.who+=self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))
        #updates the weights for the links between the input and hidden layers
        self.wih+=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),numpy.transpose(inputs))
        pass

    def query(self,inputs_list):    #query the network
        #converts inputs to 2d array
        inputs=numpy.array(inputs_list,ndmin=2).T
        #calculate signals into hidden layer
        hidden_inputs=numpy.dot(self.wih,inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs=self.activation_function(hidden_inputs)
        #calculate signals into final output layer
        final_inputs=numpy.dot(self.who,hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs=self.activation_function(final_inputs)
        return final_outputs

#now new an object
input_nodes,hidden_nodes,output_nodes=3,3,3
learning_rate=0.5

n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
print(n.query([1.0,0.5,-1.5]))

