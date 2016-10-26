# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
import sklearn.cross_validation as cross_validation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV


def readInputFiles():
    file_ptr = open('C:\\Users\\Palak\\Desktop\\Energy-Consumption.txt')
    file_ptr2 = open('C:\\Users\\Palak\\Desktop\\outsideTemp.txt')
    #put the data points from file in a list
    data1 = []
    for line in file_ptr.readlines():
        data1.append([float(x) for x in line.strip().split('\t')])
    data2 = []
    for line in file_ptr2.readlines():
        data2.append([float(x) for x in line.strip().split('\t')])
        # our X and Y axis arrays
    Y_axis = np.array(data1)/(10**10)
    Y_Energy_Consumption = np.transpose(Y_axis)
    X_outside_temp = np.array(data2)
    return Y_Energy_Consumption,X_outside_temp

def CrossValidationRegularization():
   Y_Energy_Consumption, X_outside_temp =readInputFiles()

   for degree in [2,3,4,5,6,7,8]:
       poly = PolynomialFeatures(degree)
       X_ = poly.fit_transform(X_outside_temp)
       kfold = cross_validation.KFold(len(X_outside_temp), n_folds=10)
       #X_plot_ = poly.fit_transform(X_plot)
       model = LassoCV(cv=kfold)
       model.fit(X_, Y_Energy_Consumption[6])
       print "Penalizing factor for degree", degree,"=", int(model.alpha_)
    

def plotAllGraphsAndCalcError():
    # for plotting the residual
    Y_Energy_Consumption, X_outside_temp =readInputFiles()
    dimensions=2

    #To calculate LMS and In-Sample Error
    coefficientsA =np.polyfit(X_outside_temp.ravel(),Y_Energy_Consumption[0],2)
    yfitA = np.polyval(coefficientsA,X_outside_temp.ravel())
    errorA = Y_Energy_Consumption[0]-yfitA
    #residualA = np.mean((errorA) ** 2)
    varianceA = np.var(errorA, ddof=1)
    varianceA = np.square(varianceA)
    forinSampleErrorA = 2*((dimensions/len(X_outside_temp))*varianceA)
    insampleErrorA = errorA + forinSampleErrorA
    Residual_insampleErrorA = np.mean(insampleErrorA)**2
    print "In Sample error for graph A ",Residual_insampleErrorA
    
    #To calculate LMS and In-Sample Error
    coefficientsB =np.polyfit(X_outside_temp.ravel(),Y_Energy_Consumption[1],2)
    yfitB = np.polyval(coefficientsB,X_outside_temp.ravel())
    errorB = Y_Energy_Consumption[1]-yfitB
    #residualB = np.mean((errorB) ** 2)
    varianceB = np.var(errorB, ddof=1)
    varianceB = np.square(varianceB)
    forinSampleErrorB = 2*((dimensions/len(X_outside_temp))*varianceB)
    insampleErrorB = errorB + forinSampleErrorB
    Residual_insampleErrorB = np.mean(insampleErrorB)**2
    print "In Sample error for graph B ",Residual_insampleErrorB
    
    #To calculate LMS and In-Sample Error    
    coefficientsC =np.polyfit(X_outside_temp.ravel(),Y_Energy_Consumption[2],2)
    yfitC = np.polyval(coefficientsC,X_outside_temp.ravel())
    errorC = Y_Energy_Consumption[2]-yfitC
    #residualC = np.mean((errorC) ** 2)
    varianceC = np.var(errorC, ddof=1)
    varianceC = np.square(varianceC)
    forinSampleErrorC = 2*((dimensions/len(X_outside_temp))*varianceC)
    insampleErrorC = errorC + forinSampleErrorC
    Residual_insampleErrorC = np.mean(insampleErrorC)**2
    print "In Sample error for graph C ",Residual_insampleErrorC
    
    #To calculate LMS and In-Sample Error    
    coefficientsD =np.polyfit(X_outside_temp.ravel(),Y_Energy_Consumption[3],2)
    yfitD = np.polyval(coefficientsD,X_outside_temp.ravel())
    errorD = Y_Energy_Consumption[3]-yfitD  
    #residualD = np.mean((errorD) ** 2)
    varianceD = np.var(errorD, ddof=1)
    varianceD = np.square(varianceD)
    forinSampleErrorD = 2*((dimensions/len(X_outside_temp))*varianceD)
    insampleErrorD = errorD + forinSampleErrorD
    Residual_insampleErrorD = np.mean(insampleErrorD)**2
    print "In Sample error for graph D ",Residual_insampleErrorD
   
    #To calculate LMS and In-Sample Error    
    coefficientsE =np.polyfit(X_outside_temp.ravel(),Y_Energy_Consumption[4],2)
    yfitE = np.polyval(coefficientsE,X_outside_temp.ravel())
    errorE = Y_Energy_Consumption[4]-yfitE
    #residualE = np.mean((errorE) ** 2)
    varianceE = np.var(errorE, ddof=1)
    varianceE = np.square(varianceE)
    forinSampleErrorE = 2*((dimensions/len(X_outside_temp))*varianceE)
    insampleErrorE = errorE + forinSampleErrorE
    Residual_insampleErrorE = np.mean(insampleErrorE)**2
    print "In Sample error for graph E ",Residual_insampleErrorE
    
    #To calculate LMS and In-Sample Error    
    coefficientsF =np.polyfit(X_outside_temp.ravel(),Y_Energy_Consumption[5],2)
    yfitF = np.polyval(coefficientsF,X_outside_temp.ravel())
    errorF = Y_Energy_Consumption[5]-yfitF
    #residualF = np.mean((errorF) ** 2)
    varianceF = np.var(errorF, ddof=1)
    varianceF = np.square(varianceF)
    forinSampleErrorF = 2*((dimensions/len(X_outside_temp))*varianceF)
    insampleErrorF = errorF + forinSampleErrorF
    Residual_insampleErrorF = np.mean(insampleErrorF)**2
    print "In Sample error for graph F ",Residual_insampleErrorF

    coefficientsG =np.polyfit(X_outside_temp.ravel(),Y_Energy_Consumption[6],2)
    yfitG = np.polyval(coefficientsG,X_outside_temp.ravel())
    errorG = Y_Energy_Consumption[6]-yfitG
    varianceG = np.var(errorA, ddof=1)
    varianceG = np.square(varianceG)
    forinSampleErrorG = 2*((dimensions/len(X_outside_temp))*varianceG)
    insampleErrorG = errorG + forinSampleErrorG
    Residual_insampleErrorG = np.mean(insampleErrorG)**2
    print "In Sample error for graph G ",Residual_insampleErrorG
    #residualG = np.mean((errorG) ** 2)
    
    #### To Calculate the Correlation between the Dependent Variables start
    combined = np.array([errorA, errorB, errorC, errorD,errorE,errorF,errorG])
    correlation_between_errors_=np.corrcoef(combined)[0,1]
    print "Correlation Coefficient"
    print correlation_between_errors_
    #### To Calculate the Correlation between the Dependent Variables end
    
    #Plotting All the graphs
    A=plt.scatter(X_outside_temp, Y_Energy_Consumption[0],marker='*',color='violet',label = "19.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)")
    B=plt.scatter(X_outside_temp, Y_Energy_Consumption[1],marker='*', color='indigo',label = "20.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)")
    C=plt.scatter(X_outside_temp, Y_Energy_Consumption[2],marker='*',color='blue',label = "21.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)")
    D=plt.scatter(X_outside_temp, Y_Energy_Consumption[3],marker='*',color='green',label = "22.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)")
    E=plt.scatter(X_outside_temp, Y_Energy_Consumption[4],marker='*',color='yellow',label = "23.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)")
    F=plt.scatter(X_outside_temp, Y_Energy_Consumption[5],marker='*',color='orange',label = "24.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)")
    G=plt.scatter(X_outside_temp, Y_Energy_Consumption[6],marker='*',color='red',label = "25.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)")
    plt.xlabel("Environment Site Outdoor Air DryBulb Temperature " u'\xb0' "C")
    plt.ylabel("Electricity+Gas facility [J]")
    plt.ylim((0.6),(2.2))
    plt.legend((A, B, C, D, E, F, G),
           ("19.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)",
            "20.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)", 
            "21.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)", 
            "22.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)",
            "23.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)",
            "24.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)",
            "25.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)",
            ),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=8)
           
  # For Setpoint 19.5 Degree         
def individualplotA():
    Y_Energy_Consumption, X_outside_temp =readInputFiles()
    coefficients =np.polyfit(X_outside_temp.ravel(),Y_Energy_Consumption[0],2)
    yfit = np.polyval(coefficients,X_outside_temp.ravel())
    error = Y_Energy_Consumption[0]-yfit
    #residualA = np.mean((error) ** 2)
    #Plotting the Curve Fit
    polynomial = np.poly1d(coefficients)
    ys = polynomial(X_outside_temp)
    print ys
    print "Equation of the line\n"
    print "----------------------------------"
    print polynomial
    print "----------------------------------"
    Err=plt.scatter(X_outside_temp,  error , marker='*',color='yellow')
    P=plt.scatter(X_outside_temp.ravel(),  ys, marker='.',color='red')
    # plotting the least mean squared error comment the y-limits to plot the error
    A=plt.scatter(X_outside_temp, Y_Energy_Consumption[0],marker='*',color='violet',label = "19.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)")
    plt.grid()
    plt.xlabel("Environment Site Outdoor Air DryBulb Temperature " u'\xb0' "C")
    plt.ylabel("Electricity+Gas facility [J]")
    plt.ylim((0.6),(2.2))
    plt.legend((A,P,Err),
           ("19.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)",
            "Polynomial Curve Fit",
            "Error"
            ),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=8)

# For Setpoint 20.5 Degree 
def individualplotB():
    Y_Energy_Consumption, X_outside_temp =readInputFiles()
    coefficients =np.polyfit(X_outside_temp.ravel(),Y_Energy_Consumption[1],2)
    yfit = np.polyval(coefficients,X_outside_temp.ravel())
    error = Y_Energy_Consumption[1]-yfit
    #residualB = np.mean((error) ** 2)
    polynomial = np.poly1d(coefficients)
    ys = polynomial(X_outside_temp)
    # Plotting the Curve Fit
    print "Equation of the line\n"
    print "----------------------------------"
    print polynomial
    print "----------------------------------"
    P=plt.scatter(X_outside_temp,  ys , marker='.',color='red')
    #plotting the least mean squared error comment the y-limits to plot the error
    Err=plt.scatter(X_outside_temp,  error , marker='*',color='yellow')
    A=plt.scatter(X_outside_temp, Y_Energy_Consumption[1],marker='*',color='violet',label = "19.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)")
    plt.xlabel("Environment Site Outdoor Air DryBulb Temperature " u'\xb0' "C")
    plt.ylabel("Electricity+Gas facility [J]")
    plt.ylim((0.6),(2.2))
    plt.grid()
    plt.legend((A,P,Err),
           ("20.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)",
            "Polynomial Curve Fit",
            "Error"
            ),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=8)

# For Setpoint 21.5 Degree           
def individualplotC():
    Y_Energy_Consumption, X_outside_temp =readInputFiles()
    coefficients =np.polyfit(X_outside_temp.ravel(),Y_Energy_Consumption[2],2)
    yfit = np.polyval(coefficients,X_outside_temp.ravel())
    error = Y_Energy_Consumption[2]-yfit
    #residualC = np.mean((error) ** 2)
    polynomial = np.poly1d(coefficients)
    ys = polynomial(X_outside_temp)
    print "Equation of the line\n"
    print "----------------------------------"
    print polynomial
    print "----------------------------------"
    #Plotting the Curve Fit
    P=plt.scatter(X_outside_temp.ravel(),  ys, marker='.',color='red')
    #plotting the least mean squared error comment the y-limits to plot the error
    Err=plt.scatter(X_outside_temp,  error , marker='*',color='yellow')
    A=plt.scatter(X_outside_temp, Y_Energy_Consumption[2],marker='*',color='violet',label = "19.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)")
    plt.xlabel("Environment Site Outdoor Air DryBulb Temperature " u'\xb0' "C")
    plt.ylabel("Electricity+Gas facility [J]")
    plt.ylim((0.6),(2.2))
    plt.grid()
    plt.legend((A,P,Err),
           ("21.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)",
            "Polynomial Curve Fit",
            "Error"
            ),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=8)
 
# For Setpoint 22.5 Degree           
def individualplotD():
    Y_Energy_Consumption, X_outside_temp =readInputFiles()
    coefficients =np.polyfit(X_outside_temp.ravel(),Y_Energy_Consumption[3],2)
    yfit = np.polyval(coefficients,X_outside_temp.ravel())
    error = Y_Energy_Consumption[3]-yfit
    #residualD = np.mean((error) ** 2)
    polynomial = np.poly1d(coefficients)
    ys = polynomial(X_outside_temp)
    print "Equation of the line\n"
    print "----------------------------------"
    print polynomial
    print "----------------------------------"
    #Plotting the Curve Fit
    P=plt.scatter(X_outside_temp.ravel(),  ys, marker='.',color='red')
    #plotting the least mean squared error comment the y-limits to plot the error
    Err=plt.scatter(X_outside_temp,  error , marker='*',color='yellow')
    A=plt.scatter(X_outside_temp, Y_Energy_Consumption[3],marker='*',color='violet',label = "19.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)")
    plt.xlabel("Environment Site Outdoor Air DryBulb Temperature " u'\xb0' "C")
    plt.ylabel("Electricity+Gas facility [J]")
    plt.ylim((0.6),(2.2))
    plt.grid()
    plt.legend((A,P,Err),
           ("22.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)",
            "Polynomial Curve Fit",
            "Error"
            ),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=8)

# For Setpoint 23.5 Degree            
def individualplotE():
    Y_Energy_Consumption, X_outside_temp =readInputFiles()
    coefficients =np.polyfit(X_outside_temp.ravel(),Y_Energy_Consumption[4],2)
    yfit = np.polyval(coefficients,X_outside_temp.ravel())
    error = Y_Energy_Consumption[4]-yfit
    #residualE = np.mean((error) ** 2)
    polynomial = np.poly1d(coefficients)
    ys = polynomial(X_outside_temp)
    print "Equation of the line\n"
    print "----------------------------------"
    print polynomial
    print "----------------------------------"
    # plotting the Curve Fit
    P=plt.scatter(X_outside_temp.ravel(),  ys, marker='.',color='red') 
    # plotting the least mean squared error comment the y-limits to plot the error
    Err=plt.scatter(X_outside_temp,  error , marker='*',color='yellow')
    A=plt.scatter(X_outside_temp, Y_Energy_Consumption[4],marker='*',color='violet',label = "19.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)")
    plt.xlabel("Environment Site Outdoor Air DryBulb Temperature " u'\xb0' "C")
    plt.ylabel("Electricity+Gas facility [J]")
    plt.ylim((0.6),(2.2))
    plt.grid()
    plt.legend((A,P,Err),
           ("23.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)",
            "Polynomial Curve Fit",
            "Error"
            ),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=8)
 
# For Setpoint 24.5 Degree           
def individualplotF():
    Y_Energy_Consumption, X_outside_temp =readInputFiles()
    coefficients =np.polyfit(X_outside_temp.ravel(),Y_Energy_Consumption[5],2)
    yfit = np.polyval(coefficients,X_outside_temp.ravel())
    error = Y_Energy_Consumption[5]-yfit
    #residualF = np.mean((error) ** 2)
    polynomial = np.poly1d(coefficients)
    ys = polynomial(X_outside_temp)
    print "Equation of the line\n"
    print "----------------------------------"
    print polynomial
    print "----------------------------------"
    # plotting the Curve Fit
    P=plt.scatter(X_outside_temp.ravel(),  ys, marker='.',color='red')
    # plotting the least mean squared error comment the y-limits to plot the error
    Err=plt.scatter(X_outside_temp,  error , marker='*',color='yellow')
    A=plt.scatter(X_outside_temp, Y_Energy_Consumption[5],marker='*',color='violet',label = "19.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)")
    plt.xlabel("Environment Site Outdoor Air DryBulb Temperature " u'\xb0' "C")
    plt.ylabel("Electricity+Gas facility [J]")
    plt.ylim((0.6),(2.2))
    plt.grid()
    plt.legend((A,P,Err),
           ("24.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)",
            "Polynomial Curve Fit",
            "Error"
            ),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=8)

# For Setpoint 25.5 Degree            
def individualplotG():
    Y_Energy_Consumption, X_outside_temp =readInputFiles()
    coefficients =np.polyfit(X_outside_temp.ravel(),Y_Energy_Consumption[6],2)
    yfit = np.polyval(coefficients,X_outside_temp.ravel())
    error = Y_Energy_Consumption[6]-yfit
    #residualG = np.mean((error) ** 2)
    polynomial = np.poly1d(coefficients)
    ys = polynomial(X_outside_temp)
    print "Equation of the line\n"
    print "----------------------------------"
    print polynomial
    print "----------------------------------"
    # plotting the Curve Fit
    P=plt.scatter(X_outside_temp.ravel(),  ys, marker='.',color='red')    
    # plotting the least mean squared error comment the y-limits to plot the error
    Err=plt.scatter(X_outside_temp,  error , marker='*',color='yellow')
    A=plt.scatter(X_outside_temp, Y_Energy_Consumption[6],marker='*',color='violet',label = "19.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)")
    plt.xlabel("Environment Site Outdoor Air DryBulb Temperature " u'\xb0' "C")
    plt.ylabel("Electricity+Gas facility [J]")
    plt.ylim((0.6),(2.2))
    plt.grid()
    plt.legend((A,P,Err),
           ("25.5" u'\xb0' "C +/-1.5"u'\xb0' "C (Deadband)",
            "Polynomial Curve Fit",
            "Error"
            ),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=8)
           

plotAllGraphsAndCalcError()
CrossValidationRegularization()

#######To view individual plot uncomment the functions one at a time##############

#individualplotA()
#individualplotB()
#individualplotC()
#individualplotD()
#individualplotE()
#individualplotF()
#individualplotG()







