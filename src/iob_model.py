from scipy.integrate import ode
import numpy as np

class IOB:
    def __init__(self):
        self.x1, self.x2 = 0, 0  # 초기 컴파트먼트 양 설정
        self.dx1, self.dx2 = 0, 0 # 변화율 초기화
        self.kdia = 0.025 # 변화 계수

    def step(self, insulin):
        dx1 = insulin - self.kdia * self.x1
        dx2 = self.kdia * self.x1 - self.kdia * self.x2
        self.x1, self.x2 = self.x1 + dx1, self.x2 + dx2 # euler

    def get_IOB(self):
        return self.x1+self.x2


def get_iob(data):
    IOB_calculator = IOB()
    ins = data.INS # 5분단위의 SMB
    ins_1m, iob_1m = np.zeros([len(ins)*5]), np.zeros([len(ins)*5])
    
    ins_1m = [ins[i//5] if i%5==0 else 0 for i, _ in enumerate(ins_1m)] # 1분단위로 변경
    for i, _ins in enumerate(ins_1m): # IOB 계산
        IOB_calculator.step(_ins)
        iob = IOB_calculator.get_IOB()
        iob_1m[i] = iob_1m[i] + iob

    iob_5m = iob_1m[::5]
    return iob_5m