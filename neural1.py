import numpy
import matplotlib.pyplot
#%matplotlib inline
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


#now creat an instance of neural network
input_nodes,hidden_nodes,output_nodes=784,500,10
learning_rate=0.2

n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

#load the training file
training_data_file=open("mnist_dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
#print(len(training_data_list),"\n",training_data_list[0])

#train the network
import datetime
start_time=datetime.datetime.now()
epoches=5
for e in range(epoches):
    print("Epoch :",e+1)
    for record in training_data_list:
        all_values=record.split(',')
        inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        targets=numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)

end_time=datetime.datetime.now()
print("Start:%s End:%s, Training time is:%s"%(start_time,end_time,end_time-start_time))

#load the test file
test_data_file=open("mnist_dataset/mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

"""
#Single comparation for one line
for counter in range(5):
    print("Please input the line number you wanna try:")
    line_number=int(input())
    if (line_number<1 or line_number>10000):
        print("Not in range 1-10000")
        continue

    all_values=test_data_list[line_number-1].split(',')
    result=n.query((numpy.asfarray(all_values[1:])/255.0*0.99)+0.01)
    print("The label is:",all_values[0],"\nThe neural network's answer:\n",result*100,"which is ",numpy.argmax(result))
    image_array=numpy.asfarray(all_values[1:]).reshape((28,28))
    matplotlib.pyplot.imshow(image_array,cmap='Greys',interpolation='None')
    matplotlib.pyplot.show()
"""

start_time=datetime.datetime.now()
score=0
for record in test_data_list:
    all_values=record.split(',')
    correct_label=int(all_values[0])
    inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    outputs=n.query(inputs)
    label=numpy.argmax(outputs)
    if(label==correct_label):
        score+=1
end_time=datetime.datetime.now()
print("Start:%s End:%s, Calculate time is:%s"%(start_time,end_time,end_time-start_time))
print("Epoches:",epoches,"  Learning rate:",learning_rate,"  Hidden nodes:",hidden_nodes)
print("Success rate:",(score/len(test_data_list))*100,"%")

