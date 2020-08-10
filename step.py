def	step(w, mul):
    if w <= 1/(3*mul):
        return 1/(3*mul)
    else:
        if w <= 1/(2.75*mul):
            if (w - 1/(3*mul)) < (1/(2.75*mul) - w):
                return 1/(3*mul)
            else:
                return 1/(2.75*mul)
        else:
            if w <= 1/(2.5*mul):
                if (w - 1/(2.75*mul)) < (1/(2.5*mul) - w):
                    return 1/(2.75*mul)
                else:
                    return 1/(2.5*mul)
            else:
                if w <= 1/(2.25*mul):
                    if (w - 1/(2.5*mul)) < (1/(2.25*mul) - w):
                        return 1/(2.5*mul)
                    else:
                        return 1/(2.25*mul)
                else:
                    if w <= 1/(2*mul):
                        if (w - 1/(2.25*mul)) < (1/(2*mul) - w):
                            return 1/(2.25*mul)
                        else:
                            return 1/(2*mul)
                    else:
                        if w <= 1/(1.75*mul):
                            if (w - 1/(2*mul)) < (1/(1.75*mul) - w):
                                return 1/(2*mul)
                            else:
                                return 1/(1.75*mul)
                        else:
                            if w <= 1/(1.5*mul):
                                if (w - 1/(1.75*mul)) < (1/(1.5*mul) - w):
                                    return 1/(1.75*mul)
                                else:
                                    return 1/(1.5*mul)
                            else:
                                if w <= 1/(1.25*mul):
                                    if (w - 1/(1.5*mul)) < (1/(1.25*mul) - w):
                                        return 1/(1.5*mul)
                                    else:
                                        return 1/(1.25*mul)
                                else:
                                    if w <= 1/(1*mul):
                                        if (w - 1/(1.25*mul)) < (1/(1*mul) - w):
                                            return 1/(1.25*mul)
                                        else:
                                            return 1/(1*mul)
                                    else:
                                        return 1/(1*mul)