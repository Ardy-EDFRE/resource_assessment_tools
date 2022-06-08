def overlap(x, y):
    return True if y[0] <= x[1] else False

def merge(x, y):
    return tuple((min(x[0], y[0]), max(x[1], y[1])))

def adjust_angle_forward(x):
    return tuple((x[0], x[1] + 360)) if x[1] < x[0] else x

def adjust_angle_backward(x):
    return tuple((x[0], x[1] - 360)) if x[1] > 360 else x

def condense(list_of_tuples):
    """
    Given a list of tuples, find the overlapping pairs and
    condense them into a shorter list of tuples where the start and end values
    represent the smallest and largest ends of the overlapping segments

    >>> condense([(20, 25), (18, 19), (14, 15), (7, 15), (5, 6), (2, 3), (1, 2)])
    [(1, 3), (5, 6), (7, 15), (18, 19), (20, 25)]
    >>> condense([(1, 2), (2, 11), (5, 6), (5, 10), (7, 15), (14, 15), (18, 19), (20, 25)])
    [(1, 15), (18, 19), (20, 25)]
    >>> condense([(162, 236), (49, 115), (32, 80), (93, 137), (59, 99), (290, 10), (300, 12)])
    [(32, 137), (162, 236), (290, 12)]
    """
    l = sorted(list_of_tuples)
    i = 0
    while i < len(l)-1:
        l[i] = adjust_angle_forward(l[i])
        l[i+1] = adjust_angle_forward(l[i+1])
        if overlap(l[i], l[i + 1]): # if the ith and i+1th elements overlap
            merged = merge(l[i], l[i+1]) # merge
            l[i] = merged # replace original
            del l[i+1] # delete i+1th
        else:
            i+=1 # else keep moving

        l[i] = adjust_angle_backward(l[i])

    # manual check for overlapping angles on both sides
    if overlap(l[-1], l[0]) and len(l) > 1:
        merged = tuple((l[-1][0], l[0][1]))
        l[-1] = merged
        del l[0]

    return l

def invert_angles(list_of_angles):
    """
    Given a list of condensed angle tuples, find the remaining angles that would
    complete a circle. This is used to go from "list of sectors with
    obstacles" to "list of valid sectors for terrain assessment"
    >>> invert_angles([(0, 45)])
    [(45, 360)]
    >>> invert_angles([(0, 45), (90, 120)])
    [(45, 90), (120, 360)]
    >>> invert_angles([(45, 90), (91, 120), (350, 10)])
    [(10, 45), (90, 91), (120, 350)]
    >>> invert_angles([(179, 250)])
    [250,179]
    """
    i = 0
    if len(list_of_angles) == 1:
        angle = list_of_angles[0]
##        return [adjust_angle_forward((angle[1], angle[0]))]
        angle = adjust_angle_forward((angle[1],angle[0]))

        if angle[0] >= 360:
            angle = tuple((angle[0]-360, angle[1]))
        if angle[1] >= 360:
            angle = tuple((angle[0], angle[1]-360))
  
        return [angle]

    else:
        inverted_angles = []
        while i < len(list_of_angles):
            angle = (list_of_angles[i][1], list_of_angles[(i+1) % len(list_of_angles)][0])
            inverted_angles.append(adjust_angle_forward(angle))
            i = i + 1

        ###
        inverted_angles_adjusted = []
        for angle in inverted_angles:
            if angle[0] >= 360:
                angle = tuple((angle[0]-360, angle[1]))
            if angle[1] >= 360:
                angle = tuple((angle[0], angle[1]-360))
            inverted_angles_adjusted.append(angle)

        return sorted(inverted_angles)
