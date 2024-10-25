import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quaternion as qt
import numAnalysis as na

def quat2Eul(q, offsets=(0,0,0)):
    q0, q1, q2, q3 = q.w, q.x, q.y, q.z
    phi = np.arctan2(-2*q1*q2 + 2*q0*q3, q1**2 + q0**2 - q3**2 - q2**2)
    theta = np.arcsin(2*q1*q3 + 2*q0*q2)
    psi = np.arctan2(-2*q2*q3 + 2*q0*q1, q3**2 - q2**2 - q1**2 + q0**2)
    azim = np.rad2deg(phi) + offsets[0]
    elev = -np.rad2deg(theta) + offsets[1]
    roll = np.rad2deg(psi) + offsets[2]
    return elev, azim, roll

def addEul2Df(df):
    df['Azim'] = np.rad2deg(np.arctan2(-2*df['Quat_X']*df['Quat_Y'] + 2*df['Quat_W']*df['Quat_Z'], df['Quat_X']**2 + df['Quat_W']**2 - df['Quat_Z']**2 - df['Quat_Y']**2))    
    df['Elev'] = -np.rad2deg(np.arcsin(2*df['Quat_X']*df['Quat_Z'] + 2*df['Quat_W']*df['Quat_Y']))
    df['Roll'] = np.rad2deg(np.arctan2(-2*df['Quat_Y']*df['Quat_Z'] + 2*df['Quat_W']*df['Quat_X'], df['Quat_Z']**2 - df['Quat_Y']**2 - df['Quat_X']**2 + df['Quat_W']**2))

def plotLRAccData(dfL, dfR, title="Accel on Both Arms"):
    fig, ax = plt.subplots(3, sharex=True)
    fig.suptitle(title)

    ax[0].plot(dfL["Acc_X"], label='Left Shoulder', color='blue')
    ax[0].plot(dfR["Acc_X"], label='Right Shoulder', color='red')
    ax[0].set_title('Accel X')

    ax[1].plot(dfL["Acc_Y"], color='blue')
    ax[1].plot(dfR["Acc_Y"], color='red')
    ax[1].set_title('Accel Y')
    ax[1].set(ylabel='Acceleration (m/s^2)')

    ax[2].plot(dfL["Acc_Z"], color='blue')
    ax[2].plot(dfR["Acc_Z"], color='red')
    ax[2].set_title('Accel Z')
    ax[2].set(xlabel='Sample #')
    fig.legend()
    fig.tight_layout()
    plt.show() 

def plotLRGyrData(dfL, dfR, title="Gyroscopic Data Both Arms"):
    fig, ax = plt.subplots(3, sharex=True)
    fig.suptitle(title)

    ax[0].plot(dfL["Gyr_X"], label='Left Shoulder', color='blue')
    ax[0].plot(dfR["Gyr_X"], label='Right Shoulder', color='red')
    ax[0].set_title('Gyr X')

    ax[1].plot(dfL["Gyr_Y"], color='blue')
    ax[1].plot(dfR["Gyr_Y"], color='red')
    ax[1].set_title('Gyr Y')
    ax[1].set(ylabel='Angular Velocity (rad/s)')

    ax[2].plot(dfL["Gyr_Z"], color='blue')
    ax[2].plot(dfR["Gyr_Z"], color='red')
    ax[2].set_title('Gyr Z')
    ax[2].set(xlabel='Sample #')
    fig.legend()
    fig.tight_layout()
    plt.show() 

def plotLREulData(dfL, dfR, title="Euler Angles Right Arm Comparison"):
    fig, ax = plt.subplots(3, sharex=True)
    fig.suptitle(title)

    ax[0].plot(dfL["Elev"], label='Without cane', color='blue')
    ax[0].plot(dfR["Elev"], label='With cane', color='red')
    ax[0].set_title('Pitch')

    ax[1].plot(dfL["Azim"], color='blue')
    ax[1].plot(dfR["Azim"], color='red')
    ax[1].set_title('Yaw')
    ax[1].set(ylabel='Angular velocity')

    ax[2].plot(dfL["Roll"], color='blue')
    ax[2].plot(dfR["Roll"], color='red')
    ax[2].set_title('Roll')
    fig.legend()
    fig.tight_layout()
    plt.show() 

def estimateSlant(df):
    acc = df[['Acc_X', 'Acc_Y', 'Acc_Z']].iloc[1:51].to_numpy()
    
    avgAcc = np.mean(acc, axis=0)
    #print("\nX Accel = ", avgAcc[0], ", Y Accel = ", avgAcc[1], ", Z Accel = ", avgAcc[2])
    
    thetaX = np.arctan2(avgAcc[1],np.sqrt(avgAcc[0]**2+avgAcc[2]**2))
    thetaY = np.arctan2(-avgAcc[0],avgAcc[2])
    
    thetaXDeg = np.rad2deg(thetaX)
    thetaYDeg = np.rad2deg(thetaY)
    #print("\n slant X = ", thetaXDeg,  " degrees , slant Y = ", thetaYDeg, " degrees")
    return thetaXDeg, thetaYDeg

def gyroOrient(df, rate):
    gyroData = df[['Gyr_X','Gyr_Y']].to_numpy()
    angDisp = np.cumsum(gyroData * rate, axis=0)
    return angDisp

def compFilter(accAng, gyroDisp, alpha=0.98):
    return alpha * gyroDisp + (1-alpha)*accAng

def accelAngles(df):
    accelData = df[['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
    ax, ay, az = accelData.T
    accPitch = np.arctan2(ay,np.sqrt(ax**2+az**2))
    accRoll = np.arctan2(-ax, az)
    return np.vstack((accPitch,accRoll)).T

def plotFuseOrientation(angles, title="Shoulder Orientation"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    time = np.arange(len(angles))
    pitch, roll = angles.T

    ax.plot(time, pitch, roll, label = "Pitch vs Roll")
    ax.set(xlabel='Time', ylabel='Pitch (rads)', zlabel='Roll (rads)')
    ax.set_title(title)
    ax.legend()
    plt.show()

def slantRotationMatrix(slantX, slantY):
    pitchAng = np.deg2rad(slantY)
    rollAng = np.deg2rad(slantX)

    pitchR = np.array([[np.cos(pitchAng), 0, np.sin(pitchAng)],
                      [0, 1, 0],
                      [-np.sin(pitchAng), 0, np.cos(pitchAng)]])
    rollR = np.array([[1, 0, 0],
                     [0, np.cos(rollAng), -np.sin(rollAng)],
                     [0, np.sin(rollAng), np.cos(rollAng)]])
    
    R = np.dot(rollR, pitchR)
    return R

def accelSlantCorrection(df):
    accelData = df[['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()

    slantX, slantY = estimateSlant(df)

    R = slantRotationMatrix(slantX, slantY)

    correctedAcc = []
    for a in accelData:
        correctedA = np.dot(R, a)
        correctedAcc.append(correctedA)

    return np.array(correctedAcc)

def gyrSlantCorrection(df):
    gyrData = df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()

    slantX, slantY = estimateSlant(df)

    R = slantRotationMatrix(slantX, slantY)

    correctedGyr = []
    for a in gyrData:
        correctedG = np.dot(R, a)
        correctedGyr.append(correctedG)

    return np.array(correctedGyr)

def plotAccelCorrected(dfL, dfR, title="Corrected Acceleration Right Arm Comparison"):

    print(np.mean(dfL[2:40], axis=0))

    fig, ax = plt.subplots(3, sharex=True)
    fig.suptitle(title)

    ax[0].plot(dfL[125:375,0], label='Without cane', color='blue')
    ax[0].plot(dfR[125:375,0], label='With cane', color='red')
    ax[0].set_title('Acc X')
    #ax[0].set(ylabel='Acceleration (m/s^2)')

    ax[1].plot(dfL[125:375,1], color='blue')
    ax[1].plot(dfR[125:375,1], color='red')
    ax[1].set_title('Acc Y')
    ax[1].set(ylabel='Acceleration (m/s^2)')

    ax[2].plot(dfL[125:375,2], color='blue')
    ax[2].plot(dfR[125:375,2], color='red')
    ax[2].set_title('Acc Z')
    ax[2].set(xlabel='Sample #')
    fig.legend()
    fig.tight_layout()
    plt.show()

def plotGyrCorrected(dfL, dfR, title="Corrected Gyroscope Right Arm Comparison"):

    print(np.mean(dfL[2:40], axis=0))

    fig, ax = plt.subplots(3, sharex=True)
    fig.suptitle(title)

    ax[0].plot(dfL[125:375,0], label='Without cane', color='blue')
    ax[0].plot(dfR[125:375,0], label='With cane', color='red')
    ax[0].set_title('Gyr X')
    #ax[0].set(ylabel='Angular Velocity (rad/s)')

    ax[1].plot(dfL[125:375,1], color='blue')
    ax[1].plot(dfR[125:375,1], color='red')
    ax[1].set_title('Gyr Y')
    ax[1].set(ylabel='Angular Velocity (rad/s)')

    ax[2].plot(dfL[125:375,2], color='blue')
    ax[2].plot(dfR[125:375,2], color='red')
    ax[2].set_title('Gyr Z')
    ax[2].set(xlabel='Sample #')
    fig.legend()
    fig.tight_layout()
    plt.show()

    


def main():
    rate = 1/60
    dfL = pd.read_csv(r'C:\Users\Corran\Desktop\Thesis\P1\20240910_201337\right_shoulder_20240910_201337_686.csv',skiprows=10,index_col=False)
    dfR = pd.read_csv(r'C:\Users\Corran\Desktop\Thesis\P1\20240910_202214\right_shoulder_20240910_202214_803.csv',skiprows=10,index_col=False)

    addEul2Df(dfL)
    addEul2Df(dfR)
    estimateSlant(dfL)
    
    correctAL = accelSlantCorrection(dfL)
    correctAR = accelSlantCorrection(dfR)

    correctGL = gyrSlantCorrection(dfL)
    correctGR = gyrSlantCorrection(dfR)

    plotAccelCorrected(correctAL, correctAR)

    plotGyrCorrected(correctGL, correctGR)

    #plotFuseOrientation(compFilter(accelAngles(dfL),gyroOrient(dfL, rate)))

    dfLn = dfL.loc[125:375]
    dfRn = dfR.loc[125:375]


    plotLRAccData(dfL, dfR)
    plotLRGyrData(dfL, dfR)
    plotLREulData(dfL, dfR)

    print(dfLn['Acc_Z'].corr(dfRn['Acc_Z']))



if __name__== '__main__':
    main()