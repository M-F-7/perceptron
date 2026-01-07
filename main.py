import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class Infos:
    db : pd.DataFrame
    X: pd.DataFrame
    y: pd.DataFrame
    method: str



def checkmethod(arg) -> bool:
    match (arg):
        case "parse":
            method = "parse"
        case "train":
            method = "train"
        case "predict":
            method = "predict"
        case _:
            raise ValueError("Need a real method\nExemples: <parse> <train> <predict>")
    return method


def load_data(ctx):
    mean = ctx.X.mean(axis=0)
    std = ctx.X.std(axis=0)
    np.save("mean", mean)
    np.save("std", std)

def main():
    args = sys.argv[1:]
    len_args = len(args)
    try:
        if not len_args == 1:
            raise ValueError("Wrong number of arguments")
        method = checkmethod(args[0])
        db = pd.read_csv("./data.csv")
        db.iloc[:, 1] = db.iloc[:, 1].eq("M").astype(int)
        ctx = Infos(db, db.iloc[:, :-1], db.iloc[:, -1].to_frame(), method)
        print(f"{ctx.method} ctx init")
        load_data(ctx)
    except Exception as e:
        print(f"{e}", file=sys.stderr)



if __name__ == "__main__":
    main()