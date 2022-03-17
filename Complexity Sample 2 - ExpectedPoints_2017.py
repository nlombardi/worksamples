import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import statsmodels.formula.api as smf


"""
-----------------------------------
Variable Declaration and Parameters
-----------------------------------
"""
data = pd.read_excel('PATH-TO-DATA')

# Define containers
shots, passes, crosses, dribbles, intercept, tackles, rcards, ycards, succtackles, accpass, \
    HShtPred, AShtPred, PredSht, HGoalPred, AGoalPred, HPointPred, APointPred, ExpPoints, ExpIndPts, \
    dHptsdPass, dHptsdCross, dHptsdDrib, dHptsdTack, dHptsdInt, dHptsdRCard, dHptsdYCard, \
    dAptsdPass, dAptsdCross, dAptsdDrib, dAptsdTack, dAptsdInt, dAptsdRCard, dAptsdYCard, \
    passratio, tackratio \
    = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], \
        [], [], [], [], [], [], [], []

ppts, crpts, dpts, tpts, ipts, rcpts, ycpts = \
    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# List of Coefficients
coefficients = ['', 'PassRatio', 'Crosses', 'Dribbles', 'SuccTackles', 'Interceptions', 'Red Cards', 'Yellow Cards']


"""
----------------------------------
Global functions
----------------------------------
"""
# Define function to get home/away data from the dataframe and assign it to a list for regression analysis
def appendvar(ColType, Col):
    """
    Searches the data defined above for the column and appends the result to a list
    :type ColType: list
    :type Col: str
    :return: None
    """
    for i in data[Col]:
        ColType.append(i)

def tolist(prob):

    for i in prob[0]:
        prob.append(i)
    del prob[0]

def getratio(Var1, Var2, Var3):
    for i in range(len(Var1)):
        if Var2[i] != 0:
            Var3.append(Var1[i]/Var2[i])
        else:
            Var3.append(int(0))

"""
----------------------------------
Mathematical functions
----------------------------------
"""

def calcpoints(lamh, lama, size):
    """
    Calculate the expected number of points as per the double Poisson model in
    McHale et al. (2012)
    """
    probaa, probah, probhh, probha = [], [], [], []

    # ProbHH - Top, ProbAH - Left for Home calc, ProbHA - Top, ProbAA - Left for Away calc.
    probhh.append([math.exp(-lamh)*math.pow(lamh, k)/math.factorial(k) for k in range(size)])
    probha.append([math.exp(-lama)*math.pow(lama, k)/math.factorial(k) for k in range(size)])
    probah.append([math.exp(-lama)*math.pow(lama, k)/math.factorial(k) for k in range(size)])
    probaa.append([math.exp(-lamh)*math.pow(lamh, k)/math.factorial(k) for k in range(size)])

    tolist(probhh)
    tolist(probha)
    tolist(probah)
    tolist(probaa)

    # Calculates the expected points by calculating the joint probabilities of a win and tie
    H_ExpPoints = 3*np.sum(np.sum(probah[i]*probhh[i+k] for k in range(1,size-i)) for i in range(size)) \
                  + np.sum(probah[k]*probhh[k] for k in range(size))
    A_ExpPoints = 3*np.sum(np.sum(probaa[i]*probha[i+k] for k in range(1,size-i)) for i in range(size)) \
                  + np.sum(probaa[k]*probha[k] for k in range(size))

    ExpPoints.append([H_ExpPoints, A_ExpPoints])

def calcindpoints(lamh, lama, coef, phghs, pagas, name, size):
    """
    Calculate the individual contributions to expected points
    """
    probaa, probah, probhh, probha, probhhcoef, probhacoef = [], [], [], [], [], []

    # ProbHHCoef - Top, ProbAH - Left for Home calc of (lambdaH - the derivative of lambdaCoef)
    # ProbHACoef - Top, ProbAA - Left for Away calc of (lambdaA - the derivative of lambdaCoef)
    probhhcoef.append([math.exp(-(lamh+coef*phghs))*math.pow((lamh+coef*phghs), k)/math.factorial(k) for k in range(size)])
    probhacoef.append([math.exp(-(lama+coef*pagas))*math.pow((lama+coef*pagas), k)/math.factorial(k) for k in range(size)])
    probhh.append([math.exp(-lamh)*math.pow(lamh, k)/math.factorial(k) for k in range(size)])
    probha.append([math.exp(-lama)*math.pow(lama, k)/math.factorial(k) for k in range(size)])
    probah.append([math.exp(-lama)*math.pow(lama, k)/math.factorial(k) for k in range(size)])
    probaa.append([math.exp(-lamh)*math.pow(lamh, k)/math.factorial(k) for k in range(size)])

    tolist(probhhcoef)
    tolist(probhacoef)
    tolist(probhh)
    tolist(probha)
    tolist(probah)
    tolist(probaa)

    H_ExpPointsWin = np.sum(np.sum(probah[i]*probhh[i+k] for k in range(1,size-i)) for i in range(size))
    H_ExpPointsTie = np.sum(probah[k]*probhh[k] for k in range(size))
    A_ExpPointsWin = np.sum(np.sum(probaa[i]*probha[i+k] for k in range(1,size-i)) for i in range(size))
    A_ExpPointsTie = np.sum(probaa[k]*probha[k] for k in range(size))

    H_ExpPtsCoefWin = np.sum(np.sum(probah[i]*probhhcoef[i+k] for k in range(1,size-i)) for i in range(size))
    H_ExpPtsCoefTie = np.sum(probah[k]*probhhcoef[k] for k in range(size))
    A_ExpPtsCoefWin = np.sum(np.sum(probaa[i]*probhacoef[i+k] for k in range(1,size-i)) for i in range(size))
    A_ExpPtsCoefTie = np.sum(probaa[k]*probhacoef[k] for k in range(size))

    H_ExpPointsCoef = 3*((H_ExpPtsCoefWin-H_ExpPointsWin)/coef)+((H_ExpPtsCoefTie-H_ExpPointsTie)/coef)
    A_ExpPointsCoef = 3*((A_ExpPtsCoefWin-A_ExpPointsWin)/coef)+((A_ExpPtsCoefTie-A_ExpPointsTie)/coef)

    ExpIndPts.append([name, H_ExpPointsCoef, A_ExpPointsCoef])


"""
------------------------------------
REGRESSION ANALYSIS AND CALCULATIONS
------------------------------------
"""

appendvar(shots, 'Home_Shots')
appendvar(shots, 'Away_Shots')
appendvar(passes, 'Home_Passes')
appendvar(passes, 'Away_Passes')
appendvar(accpass, 'Home_Acc. Pass')
appendvar(accpass, 'Away_Acc. Pass')
appendvar(crosses, 'Home_Crosses')
appendvar(crosses, 'Away_Crosses')
appendvar(dribbles, 'Home_Drib. Att.')
appendvar(dribbles, 'Away_Drib. Att.')
appendvar(intercept, 'Away_Int')
appendvar(intercept, 'Home_Int')
appendvar(succtackles, 'Away_Succ. Tack')
appendvar(succtackles, 'Home_Succ. Tack')
appendvar(tackles, 'Away_Tackles')
appendvar(tackles, 'Home_Tackles')
appendvar(rcards, 'Away_RedCards')
appendvar(rcards, 'Home_RedCards')
appendvar(ycards, 'Away_YelCards')
appendvar(ycards, 'Home_YelCards')

getratio(accpass, passes, passratio)
getratio(succtackles, tackles, tackratio)

# Create a new dataframe with the pulled data
data_all = pd.DataFrame({'Shots': shots, 'PassRatio': passratio, 'Crosses': crosses, 'Dribbles': dribbles,
                         'Interceptions': intercept, 'SuccTackles': succtackles, 'YCards': ycards,
                         'RCards': rcards})

# Run OLS regression
lm = smf.ols(formula='Shots ~ PassRatio + Crosses + Dribbles + SuccTackles + '
                     'Interceptions + RCards + YCards', data=data_all).fit()

# Assign results of OLS regression to new variables
lm.summary()
OLSCoef = lm.params.tolist()

# Robust regression for comparison
data_y = data_all['Shots'].values
data_x = data_all.drop('Shots', axis=1).values

lm2 = sm.RLM(data_y, data_x, M=sm.robust.norms.LeastSquares()).fit()

RobustResult = lm2.summary(yname='Shots', xname=['PassRatio', 'Crosses', 'Dribbles', 'Interceptions', 'SuccTackles',
                                                 'RCards', 'YCards'])

# Loops through all rows and multiplies the appropriate column data
# by the coefficient to predict shots and assign the results to a new column
for row in data.itertuples():
    HShtPred = int(round(OLSCoef[0] + OLSCoef[1]*(row[8] / row[7]) + OLSCoef[2]*row[12] + OLSCoef[3]*row[14] +
                         OLSCoef[4]*row[39] + OLSCoef[5]*row[40] + OLSCoef[6]*row[44] + OLSCoef[7]*row[45]))
    AShtPred = int(round(OLSCoef[0] + OLSCoef[1]*(row[30] / row[29]) + OLSCoef[2]*row[34] + OLSCoef[3]*row[36] +
                         OLSCoef[4]*row[17] + OLSCoef[5]*row[18] + OLSCoef[6]*row[22] + OLSCoef[7]*row[23]))
    PredSht.append([HShtPred, AShtPred])

# Append values to all data
PredSht = pd.DataFrame(PredSht, columns=['H_Pred_Shot', 'A_Pred_Shot'])
data['H_Pred_Shot'] = PredSht['H_Pred_Shot']
data['A_Pred_Shot'] = PredSht['A_Pred_Shot']

# Predict the number of goals based on the predicted shots
# First gets the probability of converting a shot to a goal for both home and away
P_Hg_Hs = round(np.average(data['Home_Goals'])/np.average(data['Home_Shots']),4)
P_Ag_As = round(np.average(data['Away_Goals'])/np.average(data['Away_Shots']),4)

for row in data.itertuples():
    HGoalPred.append(round(P_Hg_Hs*row[46], 1))
    AGoalPred.append(round(P_Ag_As*row[47], 1))

# Append predicted goals to all data
data['H_Pred_Goal'] = HGoalPred
data['A_Pred_Goal'] = AGoalPred

# Call the functions to calculate the expected points and individual contributions
for i in range(len(HGoalPred)):
    calcpoints(HGoalPred[i], AGoalPred[i], 100)
    for j in range(1,len(OLSCoef)):
        calcindpoints(HGoalPred[i], AGoalPred[i], OLSCoef[j], P_Hg_Hs, P_Ag_As, coefficients[j], 100)

# Append all values for each calculated difference between exp pts and the individual contribution
for i in ExpIndPts:
    index = ExpIndPts.index(i)
    if i[0] == 'PassRatio':
        ppts = ppts.append({'dHpts/dLamPass': ExpIndPts[index][1],
                            'dApts/dLamPass': ExpIndPts[index][2]}, ignore_index=True)
    elif i[0] == 'Crosses':
        crpts = crpts.append({'dHpts/dLamCrosses': ExpIndPts[index][1],
                              'dApts/dLamCrosses': ExpIndPts[index][2]}, ignore_index=True)
    elif i[0] == 'Dribbles':
        dpts = dpts.append({'dHpts/dLamDrib': ExpIndPts[index][1],
                            'dApts/dLamDrib': ExpIndPts[index][2]}, ignore_index=True)
    elif i[0] == 'SuccTackles':
        tpts = tpts.append({'dHpts/dLamTack': ExpIndPts[index][1],
                            'dApts/dLamTack': ExpIndPts[index][2]}, ignore_index=True)
    elif i[0] == 'Interceptions':
        ipts = ipts.append({'dHpts/dLamInt': ExpIndPts[index][1],
                            'dApts/dLamInt': ExpIndPts[index][2]}, ignore_index=True)
    elif i[0] == 'Red Cards':
        rcpts = rcpts.append({'dHpts/dLamRCards': ExpIndPts[index][1],
                              'dApts/dLamRCards': ExpIndPts[index][2]}, ignore_index=True)
    elif i[0] == 'Yellow Cards':
        ycpts = ycpts.append({'dHpts/dLamYCards': ExpIndPts[index][1],
                              'dApts/dLamYCards': ExpIndPts[index][2]}, ignore_index=True)
ExpIndPts = pd.concat([ppts, crpts, dpts, tpts, ipts, rcpts, ycpts], axis=1)

for i in ExpIndPts:
    for j in ExpIndPts[i]:
        if i == 'dHpts/dLamPass':
            dHptsdPass.append(j*OLSCoef[1]*P_Hg_Hs)
        if i == 'dApts/dLamPass':
            dAptsdPass.append(j*OLSCoef[1]*P_Ag_As)
        if i == 'dHpts/dLamCrosses':
            dHptsdCross.append(j*OLSCoef[2]*P_Hg_Hs)
        if i == 'dApts/dLamCrosses':
            dAptsdCross.append(j*OLSCoef[2]*P_Ag_As)
        if i == 'dHpts/dLamDrib':
            dHptsdDrib.append(j*OLSCoef[3]*P_Hg_Hs)
        if i == 'dApts/dLamDrib':
            dAptsdDrib.append(j*OLSCoef[3]*P_Ag_As)
        if i == 'dHpts/dLamTack':
            dHptsdTack.append(j*OLSCoef[4]*P_Hg_Hs)
        if i == 'dApts/dLamTack':
            dAptsdTack.append(j*OLSCoef[4]*P_Ag_As)
        if i == 'dHpts/dLamInt':
            dHptsdInt.append(j*OLSCoef[5]*P_Hg_Hs)
        if i == 'dApts/dLamInt':
            dAptsdInt.append(j*OLSCoef[5]*P_Ag_As)
        if i == 'dHpts/dLamRCards':
            dHptsdRCard.append(j*OLSCoef[6]*P_Hg_Hs)
        if i == 'dApts/dLamRCards':
            dAptsdRCard.append(j*OLSCoef[6]*P_Ag_As)
        if i == 'dHpts/dLamYCards':
            dHptsdYCard.append(j*OLSCoef[7]*P_Hg_Hs)
        if i == 'dApts/dLamYCards':
            dAptsdYCard.append(j*OLSCoef[7]*P_Ag_As)

# Assign expected points to the dataframe
ExpPoints = pd.DataFrame(ExpPoints, columns=['H_ExpPts', 'A_ExpPts'])
DerExpIndPts = pd.DataFrame({'dHpts/dPassRatio': dHptsdPass, 'dHpts/dCrosses': dHptsdCross, 'dHpts/dDrib': dHptsdDrib,
                             'dHpts/dSuccTack': dHptsdTack, 'dHpts/dInt': dHptsdInt, 'dHpts/dRCards': dHptsdRCard,
                             'dHpts/dYCards': dHptsdYCard,
                             'dApts/dPassRatio': dAptsdPass, 'dApts/dCrosses': dAptsdCross, 'dApts/dDrib': dAptsdDrib,
                             'dApts/dSuccTack': dAptsdTack, 'dApts/dInt': dAptsdInt,  'dApts/dRCards': dAptsdRCard,
                             'dApts/dYCards': dAptsdYCard})
data = pd.concat([data, ExpPoints, DerExpIndPts], axis=1)

# Export the new dataframe
data.to_excel('PATH_TO_EXPORT/data_ExpPts.xlsx')
