from abc import ABC, abstractmethod
from collections.abc import Iterable 
import os
import numpy as np
from tqdm import tqdm
try:
    import cupy as cp
    import cupyx as cpx
    glob_gpu_use = True
    print("GPU is available!")
except:
    glob_gpu_use = False
    print("GPU is NOT available! (CuPy is not imported)")

class MolPar():

    parent_dir = os.path.dirname(os.path.abspath(__file__))

    def atomic_sum(MOL,F1,F2,Df2):
        smth=0
        for i in range(len(MOL['indexes'])):
            f1=F1[i]
            f2=F2[i]
            df2=Df2[i]
            smth=smth+(f1-1j*(f2+df2))*MOL['indexes'][i]

        return smth

    def disassemble(molecula):
        molecula = molecula+' '
        elements = []
        indexes = []
        point = 0
        flag = 1
        i = 1
        eflag = 0
        while i<(len(molecula)):
            if not molecula[i].islower() and flag==1:
                elements.append("  " +molecula[point:i])
                point = i
                count = i
                flag=0
                while count<len(molecula) and (molecula[count].isdigit() or molecula[count]=='.' or molecula[count]==','):
                    count += 1
                if count == i:
                    indexes.append(1)
                else:
                    indexes.append(float(molecula[i:count]))
                point = count
                i=count
            else:
                flag = 1
                i += 1
            # print(elements)


        return {'elements':elements,'indexes':indexes}
    
    def chi0h(energy, molecula, ro):   #molecula in format like 'AlSe8O15'
        F1 = []
        F2 = []
        Co = []
        Df2 = []
        Am = []
        energy = energy*1e3

        MOL = __class__.disassemble(molecula)

        for element in MOL['elements']: #Для каждого атома нужно найти значения
                    
            with open(os.path.join(__class__.parent_dir, "chi0_chih", "f1f2_Windt.dat") , "r") as f:

                rf = f.readlines()
                l = len(rf)
                for i in range(l):
                    if rf[i].startswith("#S"):
                        rf[i] = rf[i].strip()
                        if element in rf[i] and element[-1] == rf[i][-1]: # ДОБАВИЛ ДОПОЛНИТЕЛЬНОЕ УСЛОВИЕ
                            el_l = i
                            # print(rf[i])  # ПРОВЕРКА СООТВЕТСТВИЯ ЭЛЕМЕНТОВ
                            # print(element)  # -//-
                            break

                for i in range(el_l+1,l):
                    if not rf[i].startswith("#"):
                        en_l = i
                        break

                for i in range(en_l+1,l):
                    line = rf[i].strip()
                    rb = line.find(" ")
                    en = float(line[:rb])
                    if en >= energy:
                        up_en = i
                        break
                    if int(en) == 100000:
                        print("End of list")
                        break


                p_flag = False
                if en == energy:
                    p_flag = True

                if p_flag:
                    line = rf[up_en].strip()
                    rb = line.find(" ")
                    line = line[rb:].strip()
                    rb = line.find(" ")
                    f1 = float(line[:rb])
                    line = line[rb:].strip()
                    f2 = float(line)

                else:
                    # lower energy
                    line = rf[up_en-1].strip()
                    rb = line.find(" ")
                    en_l = float(line[:rb])
                    line = line[rb:].strip()
                    rb = line.find(" ")
                    f1_l = float(line[:rb])
                    line = line[rb:].strip()
                    f2_l = float(line)

                    # upper energy
                    line = rf[up_en].strip()
                    en_u = float(line[:rb])
                    rb = line.find(" ")
                    line = line[rb:].strip()
                    rb = line.find(" ")
                    f1_u = float(line[:rb])
                    line = line[rb:].strip()
                    f2_u = float(line)

                    # interpolation f1
                    a1 = (f1_u-f1_l)/(en_u-en_l)
                    b1 = f1_u - a1*en_u
                    f1 = energy*a1+b1

                    # interpolation f2
                    a2 = (f2_u-f2_l)/(en_u-en_l)
                    b2 = f2_u - a2*en_u
                    f2 = energy*a2+b2

                F1.append(f1)
                F2.append(f2)
                    
            with open(os.path.join(__class__.parent_dir, "chi0_chih/CrossSec-Compton_McMaster.dat"), "r") as f:

                rf = f.readlines()
                l = len(rf)

                for i in range(l):
                    if rf[i].startswith("#S"):
                        rf[i] = rf[i].strip()
                        if element in rf[i] and element[-1] == rf[i][-1]:  # ДОБАВИЛ ДОПОЛНИТЕЛЬНОЕ УСЛОВИЕ
                            c=i
                            # print(rf[i])  # ПРОВЕРКА СООТВЕТСТВИЯ ЭЛЕМЕНТОВ
                            # print(element)  # -//-
                            break

                for i in range (c+1,l):
                    if not rf[i].startswith("#"):
                        c=i
                        break

                co = []
                line = rf[c]
                for i in range(3):
                    line = line.strip()
                    rb = line.find(" ")
                    co.append(float(line[:rb]))
                    line = line[rb:]
                    line = line.strip()

                co.append(float(line))
                Co.append(co)
                    
            with open(os.path.join(__class__.parent_dir, "chi0_chih/AtomicConstants.dat"), "r") as f:

                rf = f.readlines()
                l = len(rf)

                for i in range (l):
                    if rf[i].startswith("#S"):
                        rf[i] = rf[i].strip()
                        if element in rf[i] and element[-1] == rf[i][-1]:
                            c=i
                            break

                for i in range (c+1,l):
                    if not rf[i].startswith("#"):
                        c=i
                        break

                line=rf[c]
                for i in range(2):
                    line = line.strip()
                    rb = line.find(" ")
                    line = line[rb:]
                    line = line.strip()

                rb = line.find(" ")
                am = float(line[:rb])

                Am.append(am)
                
        e = energy*1e-3

        for i in range(len(Co)):  #Cчитает df2 для каждого атома
            co = Co[i]
            df2 = (1.4312e-5*e*np.exp(co[0]+co[1]*np.log(e)+co[2]*np.log(e)**2+co[3]*np.log(e)**3))
            Df2.append(df2)

        Am = np.array(Am)
        Mol = np.sum(Am*MOL['indexes']) 

        smth = __class__.atomic_sum(MOL,F1,F2,Df2) 

        chi0 = -8.3036e-4*ro/Mol/e**2*smth
        delta = abs(chi0.real/2)
        beta = abs(chi0.imag/2)

        return delta, beta

class OpticalDevice(ABC):
    
    c = 3e8 # speed of light
    gpu_use = glob_gpu_use

    @staticmethod
    def ft2d(array, dx, dy):

        if isinstance(array, np.ndarray):   
            n, m = np.shape(array)
            k, l = np.arange(0, n), np.arange(0, m)
            c_k = np.exp(1j * np.pi * k * (1-1/n))
            c_l = np.exp(1j * np.pi * l * (1-1/m))
            C_l, C_k = np.meshgrid(c_l, c_k)
            C_kl = np.multiply(C_l, C_k)
            A_n = np.exp(1j * np.pi * (1 - 1/(2*n) - n/2))
            A_m = np.exp(1j * np.pi * (1 - 1/(2*m) - m/2))

            return dx * dy * A_n * A_m * np.multiply(C_kl, np.fft.fft2(np.multiply(array, C_kl)))
    
        elif isinstance(array, cp.ndarray):
            n, m = cp.shape(array)
            k, l = cp.arange(0, n), cp.arange(0, m)
            c_k = cp.exp(1j * cp.pi * k * (1-1/n))
            c_l = cp.exp(1j * cp.pi * l * (1-1/m))
            C_l, C_k = cp.meshgrid(c_l, c_k)
            C_kl = cp.multiply(C_l, C_k)
            A_n = cp.exp(1j * np.pi * (1 - 1/(2*n) - n/2))
            A_m = cp.exp(1j * np.pi * (1 - 1/(2*m) - m/2))

            return dx * dy * A_n * A_m * np.multiply(C_kl, cpx.scipy.fftpack.fft2(cp.multiply(array, C_kl)))
        
        else:
            raise TypeError("Wrong type of input array")

    @staticmethod
    def ift2d(array, dx, dy):

        if isinstance(array, np.ndarray):
            n, m = np.shape(array)
            k, l = np.arange(0, n), np.arange(0, m)
            c_k = np.exp(-1j * np.pi * k * (1-1/n))
            c_l = np.exp(-1j * np.pi * l * (1-1/m))
            C_l, C_k = np.meshgrid(c_l, c_k)
            C_kl = np.multiply(C_l, C_k)
            A_n = np.exp(-1j * np.pi * (1 - 1/(2*n) - n/2))
            A_m = np.exp(-1j * np.pi * (1 - 1/(2*m) - m/2))

            return 1/dx * 1/dy * A_n * A_m * np.multiply(C_kl, np.fft.ifft2(np.multiply(array, C_kl)))

        elif isinstance(array, cp.ndarray):
            n, m = cp.shape(array)
            k, l = cp.arange(0, n), cp.arange(0, m)
            c_k = cp.exp(-1j * cp.pi * k * (1-1/n))
            c_l = cp.exp(-1j * cp.pi * l * (1-1/m))
            C_l, C_k = cp.meshgrid(c_l, c_k)
            C_kl = cp.multiply(C_l, C_k)
            A_n = cp.exp(-1j * np.pi * (1 - 1/(2*n) - n/2))
            A_m = cp.exp(-1j * np.pi * (1 - 1/(2*m) - m/2))

            return 1/dx * 1/dy * A_n * A_m * np.multiply(C_kl, cpx.scipy.fftpack.ifft2(cp.multiply(array, C_kl)))

    @staticmethod
    def P(x, y, z, k):
        """ Fresnel propagator """
        norm = 1
        if isinstance(x, np.ndarray):
            return norm*np.exp(1j*k*(x**2 + y**2)/(2*z))
        elif isinstance(x, cp.ndarray):
            return norm*cp.exp(1j*k*(x**2 + y**2)/(2*z))

    @staticmethod
    def fft_P(qx, qy, z, k):
        """analytical ft of Fresnel propagator """
        norm = 1
        if isinstance(qx, np.ndarray):
            return norm*np.exp(-1j*(qx**2 + qy**2)*z/(2*k))
        elif isinstance(qx, cp.ndarray):
            return norm*cp.exp(-1j*(qx**2 + qy**2)*z/(2*k))
    
    @staticmethod
    def wavelength_energy(lam=None, En=None):
        if lam == None:
            return 12.3984e-10/En, En
        elif En == None:
            return lam, 12.3984e-10/lam
        else:
            raise ValueError("wavelength / energy of light not defined")
        
    @classmethod
    def init_values(cls, new_dx, new_dy, new_Nx, new_Ny, gpu_use=glob_gpu_use):

        cls.gpu_use = gpu_use

        cls.Nx, cls.Ny = new_Nx, new_Ny
        cls.dx, cls.dy = new_dx, new_dy
        cls.dqx, cls.dqy = 2*np.pi/(cls.dx*cls.Nx), 2*np.pi/(cls.dy*cls.Ny) # step of reciprocal-space
        lbx, rbx = -(cls.Nx-1)/2, cls.Nx/2
        lby, rby = -(cls.Ny-1)/2, cls.Ny/2

        if not gpu_use:
            barrx, barry  = np.arange(lbx, rbx), np.arange(lby, rby) # base array
            cls.x, cls.y = np.meshgrid(barrx*cls.dx, barry*cls.dy)
            cls.qx, cls.qy = np.meshgrid(barrx*cls.dqx, barry*cls.dqy)
        else:
            barrx, barry  = cp.arange(lbx, rbx), cp.arange(lby, rby) # base array
            cls.x, cls.y = cp.meshgrid(barrx*cls.dx, barry*cls.dy)
            cls.qx, cls.qy = cp.meshgrid(barrx*cls.dqx, barry*cls.dqy)

    def I(self):
        return abs(self.E())**2
    
    def set_z(self, z):
        self.z = z
    
    def sv(self, arr, z, k=None):
        """ convolve arr with analytical ft of Fresnel propagator """
        k = self.k if k == None else k
        return self.ift2d(array=self.ft2d(array=arr, dx=self.dx, dy=self.dy)*self.fft_P(qx=self.qx, qy=self.qy, z=z, k=k), dx=self.dx, dy=self.dy)

    def convolve(self, arr1, arr2):
        return self.ift2d(array=self.ft2d(array=arr1, dx=self.dx, dy=self.dy)*self.ft2d(array=arr2, dx=self.dx, dy=self.dy), dx=self.dx, dy=self.dy)

    @abstractmethod
    def E(self):
        pass

class PointSource(OpticalDevice):
    
    def __init__(self, z, En=None, lam=None) -> None:
        super().__init__()
        self.lam, self.En = self.wavelength_energy(lam, En)
        self.w, self.k = 2*self.c*np.pi/self.lam, 2*np.pi/self.lam
        self.z = z
    
    def E(self):
        return super().P(x=self.x, y=self.y, z=self.z, k=self.k)

class Hole(OpticalDevice):
    
    def __init__(self, x0, y0, R, arr_start, z, lam=None, En=None):
        super().__init__()
        if lam == None:
            self.lam = 12.3984e-10/En
        elif En == None:
            self.lam = lam
        else:
            raise ValueError("wavelength / energy of light not defined")
        self.z = z
        self.arr_start = arr_start
        self.R = R
        self.x0 = x0
        self.y0 = y0
        self.w = 2*self.c*np.pi/self.lam # freq
        self.k = 2*np.pi/self.lam # wavenumber
    
    def T(self, x0=None, y0=None, x=None, y=None, R=None):
        """ "hole" transmission function """
        x = self.x
        y = self.y
        R = self.R if R == None else R
        x0 = self.x0 if x0 == None else x0
        y0 = self.y0 if y0 == None else y0

        if isinstance(x, np.ndarray):
            first_hole = (x-x0*np.ones(shape=np.shape(x)))**2 + (y-y0*np.ones(shape=np.shape(y)))**2 <= R**2
            sec_hole = (x+x0*np.ones(shape=np.shape(x)))**2 + (y+y0*np.ones(shape=np.shape(y)))**2 <= R**2
        elif isinstance(x, cp.ndarray):
            first_hole = (x-x0*cp.ones(shape=cp.shape(x)))**2 + (y-y0*cp.ones(shape=cp.shape(y)))**2 <= R**2
            sec_hole = (x+x0*cp.ones(shape=cp.shape(x)))**2 + (y+y0*cp.ones(shape=cp.shape(y)))**2 <= R**2

        return  first_hole + sec_hole
    
    def E(self):
        return self.sv(self.arr_start*self.T(), z=self.z)

class CRL2D(OpticalDevice):

    def __init__(self, z, arr_start, R, A, d, N_lens, mol, dens, lam=None, En=None):
        super().__init__()
        self.lam, self.En = self.wavelength_energy(lam, En)
        self.w, self.k = 2*self.c*np.pi/self.lam, 2*np.pi/self.lam
        self.mol, self.dens = mol, dens
        self.delta, self.betta = MolPar.chi0h(energy=self.En, molecula=self.mol, ro=self.dens)
        self.R, self.A, self.d, self.N_lens, self.copy = R, A, d, N_lens, True
        self.arr_start = arr_start
        self.z = z
        self.t_call = 1
        
    def T(self, x=None, y=None, R=None, A=None, d=None):
        """ Returns CRL's thickness"""
        x = self.x if x == None else x
        y = self.y if y == None else y
        R = self.R if R == None else R
        A = self.A if A == None else A
        d = self.d if d == None else d
        par = (x**2 + y**2)/(2*R)
        if isinstance(par, np.ndarray):
            return np.minimum(par, A**2/(8*R)) + d/2
        elif isinstance(par, cp.ndarray):
            return cp.minimum(par, A**2/(8*R)) + d/2
        
    def Trans(self, delta=None, betta=None, k=None, t_call=None):
        """ CRL-lense transmission function """
        delta = self.delta if delta == None else delta
        betta = self.betta if betta == None else betta
        k = self.k if k == None else k
        t_call = self.t_call
        if t_call == 1:
            dT = 2*self.T()
        else:
            dT = self.T() + self.T()
        if isinstance(dT, np.ndarray):
            return np.exp(-1j * k * (delta-1j*betta) * dT)
        elif isinstance(dT, cp.ndarray):
            return cp.exp(-1j * k * (delta-1j*betta) * dT)
    
    def num_wave(self, arr, N_lens=None, d=None, A=None, R=None, copy=None, k=None):
        """ arr - E on first CRL-lense's center
            returns E on last lense's center"""
        N_lens = self.N_lens if N_lens == None else N_lens
        R = self.R if R == None else R
        A = self.A if A == None else A
        d = self.d if d == None else d
        copy = self.copy if copy == None else copy
        k = self.k if k == None else k

        p = d + A**2/(4*R)
        w_f = arr

        if copy == False and N_lens > 1:
            for _ in tqdm(range(N_lens-1)):
                t1 = self.Trans()
                w_f = self.sv(arr=w_f*t1, z=p, k=k)

        elif copy == True and N_lens > 1:
            t1 = self.Trans()
            for _ in tqdm(range(N_lens-1)):
                w_f = self.sv(arr=w_f*t1, z=p, k=k)

        elif N_lens == 1:
            t1 = self.Trans()
        
        return w_f * t1
                 
    def E(self, z=None):
        """ returns E on z from last lens (i.e. z+p/2 from last lense's center)"""
        z = self.z if z == None else z 
        p = self.d + self.A**2/(4*self.R)
        wf_start = self.sv(arr=self.arr_start, z=p/2, k=self.k) # вф на середине первой линзы
        return self.sv(arr=self.num_wave(arr=wf_start), z=z+p/2, k=self.k)

    def FWHM_max(self, x_arr=None, y_arr=None):
        
        x_arr = self.x if not isinstance(x_arr, Iterable) else x_arr
        y_arr = self.I() if not isinstance(y_arr, Iterable) else y_arr
            
        max = np.max(y_arr)
        half = max/2
        i_ans = np.argwhere(np.diff(np.sign(y_arr - half))).flatten()
        
        if len(i_ans) == 2:
            x0, y0 = x_arr[i_ans[0]], y_arr[i_ans[0]]
            x1, y1 = x_arr[i_ans[1]], y_arr[i_ans[1]]

            xc0, yc0 = x_arr[i_ans[0]+1], y_arr[i_ans[0]+1]
            xc1, yc1 = x_arr[i_ans[1]+1], y_arr[i_ans[1]+1]

            a0 = (y0-yc0)/(x0-xc0)
            b0 = (y0*xc0-yc0*x0)/(xc0-x0)

            a1 = (y1-yc1)/(x1-xc1)
            b1 = (y1*xc1-yc1*x1)/(xc1-x1)

            x_search_0 = (half-b0)/a0
            x_search_1 = (half-b1)/a1
            
            return x_search_1 - x_search_0, max
        else:
            return np.nan, max

    def focus_beam(self, eps, n_dots, focus=None):
        """n_dots - нечетное число, если хотим посчитать в фокусе"""
        focus = self.focus() if focus == None else focus
        # x_eps = self.x[-1] if x_eps == None else x_eps
        x_arr = self.x
        y_arr = self.y
        n_dots = n_dots - 1
        dz = 2 * eps / n_dots
        E_arr = self.E(z=0)
        # print(np.shape(E_arr))
        if isinstance(E_arr, np.ndarray):
            z_arr = np.arange(focus-eps, focus+eps, dz)
            arr_3d = np.empty(shape=(len(x_arr), len(y_arr), len(z_arr)))
            for i,z in enumerate(z_arr):
                wfs = self.sv_cpu(arr=E_arr, z=z, k=self.k)
                # wfs = wfs[abs(x_arr) <= x_eps]
                arr_3d[i] = abs(wfs)**2
            # max_peak = np.max(arr_3d)
        elif isinstance(E_arr, cp.ndarray):
            z_arr = cp.arange(focus-eps, focus+eps, dz)
            n, m = cp.shape(E_arr)
            arr_3d = cp.empty(shape=(len(z_arr), n, m))
            # print(cp.shape(arr_3d))
            for i,z in enumerate(z_arr):
                wfs = self.sv_gpu(arr=E_arr, z=focus, k=self.k)
                # wfs = wfs[x_arr**2 + y_arr**2 <= x_eps**2]
                arr_3d[i] = abs(wfs)**2
            # print(cp.shape(wfs))
            # max_peak = cp.max(arr_3d)

        # z_arr = z_arr[z_arr > 0]
        
        # lst = []
        # for z in z_arr:
        #     wfs = self.sv_cpu(arr=E_arr, z=z, k=self.k)
        #     wfs = wfs[abs(x_arr) <= x_eps]
        #     lst.append(abs(wfs)**2)
        # x_arr = x_arr[abs(x_arr) <= x_eps]
        # lst = np.array(lst)
        # max_peak = np.max()
        # ind = np.unravel_index(np.argmax(arr_3d, axis=None), arr_3d.shape)
        # z_max = z_arr[ind[0]]
        # x_max = x_arr[ind[1]]
        # y_arr = arr_3d[ind[0]]
        # half = max_peak / 2
        # i_ans = np.argwhere(np.diff(np.sign(y_arr - half))).flatten()
    
        # if len(i_ans) == 2:
        #     x0, y0 = x_arr[i_ans[0]], y_arr[i_ans[0]]
        #     x1, y1 = x_arr[i_ans[1]], y_arr[i_ans[1]]

        #     xc0, yc0 = x_arr[i_ans[0]+1], y_arr[i_ans[0]+1]
        #     xc1, yc1 = x_arr[i_ans[1]+1], y_arr[i_ans[1]+1]

        #     a0 = (y0-yc0)/(x0-xc0)
        #     b0 = (y0*xc0-yc0*x0)/(xc0-x0)

        #     a1 = (y1-yc1)/(x1-xc1)
        #     b1 = (y1*xc1-yc1*x1)/(xc1-x1)

        #     x_search_0 = (half-b0)/a0
        #     x_search_1 = (half-b1)/a1
        #     FWHM_eff = x_search_1 - x_search_0
        # else:
        #     FWHM_eff = np.nan

        return z_arr, arr_3d

    def focus(self, N_lens=None, d=None, A=None, R=None, delta=None):
        N_lens = self.N_lens if N_lens == None else N_lens
        R = self.R if R == None else R
        A = self.A if A == None else A
        d = self.d if d == None else d
        delta = self.delta if delta == None else delta
        p = d+A**2/(4*R)
        Lc = np.sqrt(p*R/(2*delta))
        u = N_lens*p/Lc

        return Lc*np.cos(u)/np.sin(u)

    def image_prop(self, z0, z1, N_lens=None, delta=None, betta=None, d=None, A=None, R=None, k=None, x=None):
        """ image propagator """

        N_lens = self.N_lens if N_lens == None else N_lens
        x = self.x if not isinstance(x, Iterable) else x
        R = self.R if R == None else R
        A = self.A if A == None else A
        d = self.d if d == None else d
        k = self.k if k == None else k
        delta = self.delta if delta == None else delta  
        betta = self.betta if betta == None else betta

        r1 = z1
        p = d+A**2/(4*R)
        eta = delta-1j*(betta)
        zc = np.sqrt(p*R/(2*(eta)))
        L = N_lens*p
        Cl = np.cos(L/zc)
        Sl = np.sin(L/zc)
        # Ci=Cl-Sl*z1/zc
        C0 = Cl-Sl*z0/zc
        r0 = z0
        rg = (r1+r0)*Cl+(zc-r1*r0/zc)*Sl
        absor_param = np.exp(-1j*k*eta*N_lens*d)
        # absor_param = 1
        return np.sqrt(r0/rg)*(np.exp(1j*k*(C0*x**2)/(2*rg)))*absor_param
    
    def analytic_solution_CRL(self, z0, z1, N_lens=None, delta=None, betta=None, d=None, A=None, R=None, k=None, x=None):
        
        N_lens = self.N_lens if N_lens == None else N_lens
        x = self.x if not isinstance(x, Iterable) else x
        R = self.R if R == None else R
        A = self.A if A == None else A
        d = self.d if d == None else d
        k = self.k if k == None else k
        delta = self.delta if delta == None else delta  
        betta = self.betta if betta == None else betta

        return abs(self.image_prop(z0=z0, z1=z1, N_lens=N_lens, delta=delta, betta=betta, d=d, A=A, R=R, k=k, x=x))**2

class CRL1D(CRL2D):

    def __init__(self, z, arr_start, R, A, d, N_lens, mol, dens, lt, lam=None, En=None):
        self.lam, self.En = self.wavelength_energy(lam, En)
        self.w, self.k = 2*self.c*np.pi/self.lam, 2*np.pi/self.lam
        self.mol, self.dens = mol, dens
        self.delta, self.betta = MolPar.chi0h(energy=self.En, molecula=self.mol, ro=self.dens)
        self.R, self.A, self.d, self.N_lens, self.copy = R, A, d, N_lens, True
        self.arr_start = arr_start
        self.z = z
        self.lt = lt
        self.t_call = 1

    def T(self, x=None, y=None, R=None, A=None, d=None, lt=None):
        """ Returns CRL's thickness"""
        x = self.x if x == None else x
        y = self.y if y == None else y
        R = self.R if R == None else R
        A = self.A if A == None else A
        d = self.d if d == None else d
        lt = self.lt if lt == None else lt
        x_l = -lt/2
        x_r = lt/2
        mask = (x_l < x) & (x < x_r)
        anti_mask =  (x <= x_l) | (x_r <= x) 
        h_max = A**2/(8*R)
        n = 12
        alpha = A**2/(8*R)*x_r**(-n)
        if isinstance(x, np.ndarray):
            y = np.multiply(y, mask)
            par = y**2/(2*R) + anti_mask*h_max + alpha*x**n
            return np.minimum(par, h_max) + d/2
        elif isinstance(x, cp.ndarray):
            y = cp.multiply(y, mask)
            par = y**2/(2*R) + anti_mask*h_max + alpha*x**n
            return cp.minimum(par, h_max) + d/2

class CRL2Dm(CRL2D):

    @staticmethod
    def wm_2d(x, y, sig, D, M, phase):
        L = x[0][-1] - x[0][0]
        Ls = x[0][1] - x[0][0]
        x = x + 1
        y = y + 1
        gamma = 1.5
        if isinstance(x, np.ndarray):
            C = L * (sig/L)**(D-2) * (np.log(gamma)/M)**0.5
            n_max = int(np.log(L/Ls)/np.log(gamma))
            A = np.zeros(np.shape(x))
            for m in range(1, M):
                for n in range(0, int(n_max)+1):
                    phase = np.random.randint(0, 2*np.pi)
                    A += gamma**(n*(D-3))*(np.cos(phase) - np.cos(2 * np.pi * np.sqrt(x**2+y**2) * (gamma**n) * (1/L) * np.cos(np.arctan(y/x) - np.pi*m/M) + phase))
            WM = C * A
            WM_scaled = sig*(WM-np.mean(WM))/np.std(WM)

        if isinstance(x, cp.ndarray):
            C = L * (sig/L)**(D-2) * (cp.log(gamma)/M)**0.5
            n_max = int(cp.log(L/Ls)/cp.log(gamma))
            A = cp.zeros(cp.shape(x))
            for m in range(1, M):
                for n in range(0, int(n_max)+1):
                    phase = cp.random.randint(0, 2*cp.pi)
                    A += gamma**(n*(D-3))*(cp.cos(phase) - cp.cos(2 * cp.pi * cp.sqrt(x**2+y**2) * (gamma**n) * (1/L) * cp.cos(cp.arctan(y/x) - cp.pi*m/M) + phase))
            WM = C * A
            WM_scaled = sig*(WM-cp.mean(WM))/cp.std(WM)
        
        return WM_scaled
    
    def __init__(self, z, arr_start, R, A, d, N_lens, mol, dens, copy, b, m, arr_s, arr_w, arr_phase, lam=None, En=None):
        self.lam, self.En = self.wavelength_energy(lam, En)
        self.w, self.k = 2*self.c*np.pi/self.lam, 2*np.pi/self.lam
        self.mol, self.dens = mol, dens
        self.delta, self.betta = MolPar.chi0h(energy=self.En, molecula=self.mol, ro=self.dens)
        self.R, self.A, self.d, self.N_lens, self.copy = R, A, d, N_lens, copy
        self.arr_start = arr_start
        self.z = z
        self.b, self.m = b, m
        self.t_call = 1 if copy == True else 2

        if (copy == True and len(arr_w) == 2) or (copy == False and len(arr_w) == self.N_lens*2):
            self.arr_w = arr_w
        if (copy == True and len(arr_s) == 2) or (copy == False and len(arr_s) == self.N_lens*2):
            self.arr_s = arr_s
        if (copy == True and len(arr_phase) == 2) or (copy == False and len(arr_phase) == self.N_lens*2):
            self.arr_phase = arr_phase
    
    def T(self, x=None, y=None, R=None, A=None, d=None, b=None, m=None):
        """returns lens thickness considering rough surface"""

        x = self.x if not isinstance(x, Iterable) else x
        y = self.y if not isinstance(y, Iterable) else y
        R = self.R if R == None else R
        A = self.A if A == None else A
        d = self.d if d == None else d
        b = self.b if b == None else b
        m = self.m if m == None else m

        par = (x**2 + y**2)/(2*R) + self.wm_2d(x=x, y=y, sig=b, D=m, M=10, phase=0)

        if isinstance(par, np.ndarray):
            return np.minimum(par, A**2/(8*R)) + d/2
        elif isinstance(par, cp.ndarray):
            return cp.minimum(par, A**2/(8*R)) + d/2

class CRL1Dm(CRL1D):

    @staticmethod
    def wm_1d(x, phase, b, m):
        length = x[-1][0] - x[0][0]
        arr_step = x[1][0] - x[0][0]
        arr = x + 1
        D = m
        gamma = 1.5
        sigma = b
        if isinstance(x, np.ndarray):
            n_min = np.log(1/(length)) / np.log(gamma)
            n_max = np.log(1/arr_step) / np.log(gamma)
            Rx = np.zeros(np.shape(arr))
            for i in range(int(n_min)+1, int(n_max)):
                Rx += np.cos(2*np.pi*gamma**i*arr+phase)/(gamma**((2-D)*i))
            Rx = sigma*(Rx-np.mean(Rx))/np.std(Rx)
        
        if isinstance(x, cp.ndarray):
            n_min = cp.log(1/(length)) / cp.log(gamma)
            n_max = cp.log(1/arr_step) / cp.log(gamma)
            Rx = cp.zeros(cp.shape(arr))
            for i in range(int(n_min)+1, int(n_max)):
                Rx += cp.cos(2*cp.pi*gamma**i*arr+phase)/(gamma**((2-D)*i))
            Rx = sigma*(Rx-cp.mean(Rx))/cp.std(Rx)
        return Rx

    def __init__(self, z, arr_start, R, A, d, N_lens, mol, dens, copy, bx, by, mx, my, arr_s, arr_w, arr_phase, lt, lam=None, En=None):
        self.lam, self.En = self.wavelength_energy(lam, En)
        self.w, self.k = 2*self.c*np.pi/self.lam, 2*np.pi/self.lam
        self.mol, self.dens = mol, dens
        self.delta, self.betta = MolPar.chi0h(energy=self.En, molecula=self.mol, ro=self.dens)
        self.R, self.A, self.d, self.N_lens, self.copy = R, A, d, N_lens, copy
        self.arr_start = arr_start
        self.z = z
        self.lt = lt
        self.bx, self.by, self.mx, self.my = bx, by, mx, my
        self.t_call = 1 if copy == True else 2

        if (copy == True and len(arr_w) == 2) or (copy == False and len(arr_w) == self.N_lens*2):
            self.arr_w = arr_w
        if (copy == True and len(arr_s) == 2) or (copy == False and len(arr_s) == self.N_lens*2):
            self.arr_s = arr_s
        if (copy == True and len(arr_phase) == 2) or (copy == False and len(arr_phase) == self.N_lens*2):
            self.arr_phase = arr_phase

    # def T(self, x=None, y=None, R=None, A=None, d=None, mx=None, my=None, bx=None, by=None, lt=None):
    #     """ Returns CRL's thickness"""
    #     x = self.x if x == None else x
    #     y = self.y if y == None else y
    #     R = self.R if R == None else R
    #     A = self.A if A == None else A
    #     d = self.d if d == None else d
    #     bx = self.bx if bx == None else bx
    #     by = self.by if by == None else by
    #     mx = self.mx if mx == None else mx
    #     my = self.my if my == None else my
    #     lt = self.lt if lt == None else lt
    #     x_l = -lt/2
    #     x_r = lt/2
    #     mask = (x_l < x) & (x < x_r)
    #     anti_mask = (x_l >= x) | (x >= x_r)
    #     y = cp.multiply(y, mask)
    #     n = 6
    #     alpha = A**2/(8*R)*x_r**(-n)
    #     par = y**2/(2*R) + anti_mask*100 + bx*(abs(np.sin(x*mx + np.random.randint(0, 2*np.pi))) - 2/np.pi) + self.wm_1d(x=x, phase=0, b=by, m=my)
    #     par = y**2/(2*R) + anti_mask*100 + bx*(abs(np.sin(x*mx + np.random.randint(0, 2*np.pi))) - 2/np.pi) + self.wm_1d(x=x, phase=0, b=by, m=my)
    #     if isinstance(x, np.ndarray):
    #         return np.minimum(par, A**2/(8*R)) + d/2
    #     elif isinstance(x, cp.ndarray):
    #         return cp.minimum(par, A**2/(8*R)) + d/2
        
    def T(self, x=None, y=None, R=None, A=None, d=None, mx=None, my=None, bx=None, by=None, lt=None):
        """ Returns CRL's thickness"""
        x = self.x if x == None else x
        y = self.y if y == None else y
        R = self.R if R == None else R
        A = self.A if A == None else A
        d = self.d if d == None else d
        lt = self.lt if lt == None else lt
        bx = self.bx if bx == None else bx
        by = self.by if by == None else by
        mx = self.mx if mx == None else mx
        my = self.my if my == None else my

        x_l = -lt/2
        x_r = lt/2
        mask = (x_l < x) & (x < x_r)
        anti_mask =  (x <= x_l) | (x_r <= x) 
        h_max = A**2/(8*R)
        n = 6
        alpha = A**2/(8*R)*x_r**(-n)
        if isinstance(x, np.ndarray):
            y_s = np.multiply(y, mask)
            par = y_s**2/(2*R) + anti_mask*h_max + alpha*x**n + self.wm_1d(x=y, phase=0, b=by, m=my) + bx*np.sin(x*mx + np.random.rand()*2*np.pi)
            return np.minimum(par, h_max) + d/2
        elif isinstance(x, cp.ndarray):
            y_s = cp.multiply(y, mask)
            par = y_s**2/(2*R) + anti_mask*h_max + alpha*x**n + self.wm_1d(x=y, phase=np.random.rand()*2*np.pi, b=by, m=my) + bx*abs(cp.sin(x*mx + np.random.rand()*2*np.pi) - 2/np.pi)
            return cp.minimum(par, h_max) + d/2

if __name__ == "__main__":

    """Добавление библиотеки в код"""
    # import opticaldevicelib as od
    OpticalDevice.init_values(new_dx=1e-6, new_dy=1e-6, new_Nx=2**9, new_Ny=2**9, gpu_use=True)

    """Способ инициализации точечного источника c энергией En и распр. ВФ на расстоянии z от него"""
    p = PointSource(z=100, En=10)
    E_arr = p.E()
    h = Hole(z=50, x0=2e-5, y0=0, arr_start=E_arr, R=1e-5, lam=p.lam)
    E_arr2 = h.E()
    


    # N_lens_global = 1
    # crl = CRL(lam=p.lam, arr_start=E_arr,\
    #             R=6.25e-6, A=50e-6, d=2e-6, N_lens=N_lens_global, z=0,\
    #                 molecula="Si", density=2.33, Flen=0, gap=0)
    # focus = crl.focus()
    # crl.set_z(z=focus)

    # import plotly.graph_objects as go

    # z_arr, arr_3d = crl.focus_beam(eps=1e-7, n_dots=21)
    # x_ = p.x.get()[0]
    # y_ = p.y.get()[:, 0]
    # z_ = z_arr.get()
    # X, Y, Z = np.meshgrid(x_, y_, z_)
    # values = arr_3d.get().T
    # fig = go.Figure(data=go.Volume(
    # x=X.flatten()*1e6,
    # y=Y.flatten()*1e6,
    # z=Z.flatten(),
    # value=values.flatten(),
    # isomin=0.1,
    # isomax=0.5,
    # opacity=0.1, # needs to be small to see through all surfaces
    # surface_count=21, # needs to be a large number for good volume rendering
    # ))
    # fig.show()

    # h2 = Hole(z=0, x0=-1e-5, y0=-1e-4, arr_start=E_arr, D=1e-4, lam=p.lam)

    # import matplotlib.pyplot as plt
    # from matplotlib import cm
    # from matplotlib.widgets import Slider


    # fig = plt.figure()

    # ax1 = fig.add_axes([0, 0, 1, 0.8])
    # ax2 = fig.add_axes([0.1, 0.85, 0.8, 0.1])

    # s = Slider(ax = ax2, label = 'value', valmin = 0, valmax = 5, valinit = 2)

    # N = 100

    # X = np.linspace(0, 20, N)
    # Y = np.linspace(0, 20, N)
    # x, y = np.meshgrid(X, Y)
    # z = np.sin(x) + np.sin(y)

    # def update(val):
    #     value = s.val
    #     ax1.cla()
    #     ax1.pcolormesh(x, y, z + value)
    #     # ax1.set_zlim(-2, 7)

    # s.on_changed(update)
    # update(0)

    # plt.show()


    # fig1, ax1 = plt.subplots(1,1)
    # # pcol = plt.pcolor(p.x.get(), p.y.get(), p.E().imag.get())
    # # pcol = plt.pcolor(p.x.get(), p.y.get(), crl.T().get())
    # pcol = plt.pcolor(p.x.get(), p.y.get(), crl.I(z=focus).get())
    # # ax1.set_xlim(-5, 5)
    # # ax1.set_ylim(-5, 5)
    # plt.colorbar(pcol)

    # plt.show()

    # z_arr, arr_3d = crl.focus_beam(eps=1e-7, n_dots=11)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # x_ = p.x.get()[0]
    # y_ = p.y.get()[:, 0]
    # z_ = z_arr.get()
    # # c = arr_3d.get()
    # x, y, z = np.meshgrid(x_, y_, z_, indexing='xy')


    # assert np.all(x[:,0,0] == x_)
    # assert np.all(y[0,:,0] == y_)
    # assert np.all(z[0,0,:] == z_)
    # print(f"x_shape = {x}")


    # fig.set_facecolor('black')
    # ax.set_facecolor('black') 
    # ax.grid(False) 
    # ax.w_xaxis.pane.fill = False
    # ax.w_yaxis.pane.fill = False
    # ax.w_zaxis.pane.fill = False
    # ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    # ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    # ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    # c = arr_3d.get()

    # img = ax.scatter(x, y, z, alpha=c.T/np.max(c))
    # # fig.colorbar(img)
    # plt.show()



    