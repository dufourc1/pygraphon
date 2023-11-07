def graph_celtic(x, y, par, p):
    if (((x-par[0])/(par[2]/3))**2 + (y-par[1])/par[2]**2 < 1) and (((x-par[0])/par[2])**2 + (y-par[1])/(par[2]/3)**2 < 1):
        return p[0]
    elif (((x-par[1])/(par[2]/3))**2 + (y-par[0])/par[2]**2 < 1) and (((x-par[1])/par[2])**2 + (y-par[0])/(par[2]/3)**2 < 1):
        return p[0]

    elif (((x-par[0])/(par[2]/3))**2 + (y-par[1])/par[2]**2 < 1) or (((x-par[0])/par[2])**2 + (y-par[1])/(par[2]/3)**2 < 1):
        return p[1]
    elif (((x-par[1])/(par[2]/3))**2 + (y-par[0])/par[2]**2 < 1) or (((x-par[1])/par[2])**2 + (y-par[0])/(par[2]/3)**2 < 1):
        return p[1]

    elif (x-par[0])**2 + (y-par[1])**2 < par[2]**2:
        return p[2]
    elif (x-par[1])**2 + (y-par[0])**2 < par[2]**2:
        return p[2]
    else:
        return p[3]

def graph_concentric_circle(x, y, par, p):
    if (x-par[0])**2 + (y-par[1])**2 < par[2]**2:
        return p[0]
    elif (x-par[1])**2 + (y-par[0])**2 < par[2]**2:
        return p[0]

    elif (x-par[0])**2 + (y-par[1])**2 < par[3]**2:
        return p[1]
    elif (x-par[1])**2 + (y-par[0])**2 < par[3]**2:
        return p[1]

    elif (x-par[0])**2 + (y-par[1])**2 < par[4]**2:
        return p[2]
    elif (x-par[1])**2 + (y-par[0])**2 < par[4]**2:
        return p[2]

    elif (x-par[0])**2 + (y-par[1])**2 < par[5]**2:
        return p[3]
    elif (x-par[1])**2 + (y-par[0])**2 < par[5]**2:
        return p[3]
    else:
        return p[4]

def graph2_celtic(x, y, par, p):
    if ((((x - par[0]) / (par[2] / 3))**2 + (y - par[1]) / par[2]**2) < 1 and (((x - par[0]) / par[2])**2 + (y - par[1]) / (par[2] / 3)**2) < 1):
        return p[0]
    elif ((((x - par[1]) / (par[2] / 3))**2 + (y - par[0]) / par[2]**2) < 1 and (((x - par[1]) / par[2])**2 + (y - par[0]) / (par[2] / 3)**2) < 1):
        return p[0]

    elif ((((x - par[0]) / (par[2] / 3))**2 + (y - par[1]) / par[2]**2) < 1 or (((x - par[0]) / par[2])**2 + (y - par[1]) / (par[2] / 3)**2) < 1):
        return p[1]
    elif ((((x - par[1]) / (par[2] / 3))**2 + (y - par[0]) / par[2]**2) < 1 or (((x - par[1]) / par[2])**2 + (y - par[0]) / (par[2] / 3)**2) < 1):
        return p[1]

    elif (x - par[0])**2 + (y - par[1])**2 < par[2]**2:
        return p[2]
    elif (x - par[1])**2 + (y - par[0])**2 < par[2]**2:
        return p[2]
    elif (x - par[3])**2 + (y - par[4])**2 < par[5]**2:
        return p[3]
    elif (x - 1 + par[3])**2 + (y - 1 + par[4])**2 < par[5]**2:
        return p[3]
    else:
        return p[4]

def graph2_concentric_circle(x, y, par, p):
    if (x - par[0])**2 + (y - par[1])**2 < par[2]**2:
        return p[0]
    elif (x - par[1])**2 + (y - par[0])**2 < par[2]**2:
        return p[0]

    elif (x - par[0])**2 + (y - par[1])**2 < par[3]**2:
        return p[1]
    elif (x - par[1])**2 + (y - par[0])**2 < par[3]**2:
        return p[1]

    # ... continue the pattern for the other conditions ...
    
    elif (x - par[6])**2 + (y - par[7])**2 < par[8]**2:
        return p[4]
    elif (x - 1 + par[7])**2 + (y - 1 + par[6])**2 < par[8]**2:
        return p[4]
    elif (x - par[6])**2 + (y - par[7])**2 < par[9]**2:
        return p[5]
    elif (x - 1 + par[7])**2 + (y - 1 + par[6])**2 < par[9]**2:
        return p[5]
    
    # ... continue the pattern for the remaining conditions ...

    else:
        return p[9]
