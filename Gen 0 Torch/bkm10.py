import torch
import torch.nn as nn


class BKM10(nn.Module):

    def __init__(self):

        super().__init__()

        self.register_buffer("PI", torch.tensor(
            3.141592653589793, dtype=torch.float32))
        self.register_buffer("ALPHA", torch.tensor(
            1/137.0359998, dtype=torch.float32))
        self.register_buffer("M2", torch.tensor(
            0.93827208816*0.93827208816, dtype=torch.float32))
        self.register_buffer("GeV_to_nb", torch.tensor(
            0.389379e6, dtype=torch.float32))

    def compute_kinematics(self, kins):
        """
        Computes intermediate kinematic quantities.

        Parameters:
            kins (sequence of float, shape (5,)) : Kinematic variables [k, Q², x_b, t, φ] where
            - k : incoming lepton energy in the target rest frame (GeV)
            - Q² : virtuality of the photon (GeV²)
            - x_b : Bjorken x (dimensionless)
            - t   : negative squared momentum transfer (GeV²)
            - φ   : azimuthal angle between leptonic and hadronic planes (radians)
            helc (int, shape(1,)) : Beam helicity, can be -1, 0 or 1
            cffs ((sequence of float, shape (4,))) : 
                Compton form factors [Re(H), Re(H̃), Re(E), |Amplitude_DVCS|²].

        Returns:
            tuple : (Q2, x, t, phi, e2, y, t_min)
        """

        k, Q2, x, t, phi = kins.T

        # BKM 2010 : Eq (1.2)
        e2 = 4.*x*x*self.M2/Q2

        # Definition of lepton energy loss. Maybe from DIS theory?
        y = torch.sqrt(Q2/e2)/k

        # BKM 2002 : Eq (31)
        t_min = -Q2*(2.*(1.-x)*(1.-torch.sqrt(1+e2))+e2)/(4.*x*(1.-x)+e2)

        # BKM 2010 : EQ (2.5) and BKM 2002 : EQ (30)?
        K_tilde = torch.sqrt((t_min-t)*((1.-x)*torch.sqrt(1.+e2) + (t-t_min)*(e2+4.*x*(1.-x))/(4.*Q2)) *
                             (1.-y-y*y*e2/4.)/(1.-y+y*y*e2/4.))

        # K_tilde = tf.sqrt((t_min-t)*((1.-x)*tf.sqrt(1.+e2) + (t_min-t)*(e2+4.*x*(1.-x))/(4.*Q2)))

        # BKM 2010 : Below Eq (2.21)
        K = torch.sqrt((1.-y+e2*y*y*0.25)/Q2)*K_tilde

        # BKM 2002 : Eq (29)
        kDelta = -(Q2/(2.*y*(1.+e2))) * (1. + 2.*K*torch.cos(phi) -
                                         (t/Q2)*(1.-x*(2.-y)+0.5*y*e2) + 0.5*y*e2)

        F1, F2 = self.form_factors(t)

        # BKM 2002 : Eq (28)
        P1 = 1. + (2.*kDelta)/Q2
        P2 = (-2.*kDelta+t)/Q2

        # BKM 2010 : prefactor in Eq (1.1)
        dsig_prefactor = self.ALPHA*self.ALPHA*self.ALPHA * \
            x*y*y/(8.*self.PI*Q2*Q2*torch.sqrt(1.+e2))

        return Q2, x, t, phi, e2, y, t_min, K_tilde, K, F1, F2, P1, P2, dsig_prefactor

    def diff_cross(self, kins, cffs, helc=0):
        """
        Computes the differential cross section for the BKM formalism in twist-2 for now,
        based on Eqn (1.1) BKM 2010 paper.

        Parameters:
            kins (sequence of float, shape (5,)) : Kinematic variables [k, Q², x_b, t, φ] where
            - k : incoming lepton energy in the target rest frame (GeV)
            - Q² : virtuality of the photon (GeV²)
            - x_b : Bjorken x (dimensionless)
            - t   : negative squared momentum transfer (GeV²)
            - φ   : azimuthal angle between leptonic and hadronic planes (radians)
            helc (int, shape(1,)) : Beam helicity, can be -1, 0 or 1
            cffs ((sequence of float, shape (4,))) : 
                Compton form factors [Re(H), Re(H̃), Re(E), |Amplitude_DVCS|²].

        Returns:
            d_sigma : float
                Differential cross section d⁴σ/(dQ² dt dx_b dΦ), in nb.
        """

        Q2, x, t, phi, e2, y, t_min, K_tilde, K, F1, F2, P1, P2, dsig_prefactor = \
            self.compute_kinematics(kins)

        ReH, ReHt, ReE, T_DVCS_2 = cffs.T
        ImH = torch.tensor(0.0, dtype=torch.float32, device=Q2.device)
        ImHt = torch.tensor(0.0, dtype=torch.float32, device=Q2.device)
        ImE = torch.tensor(0.0, dtype=torch.float32, device=Q2.device)

        T_BH_2 = self.compute_BH_amplitude(
            Q2, t, x, phi, e2, y, K, P1, P2, F1, F2, dsig_prefactor)

        I = self.compute_Interference(Q2, t, x, phi, e2, y, K, t_min, K_tilde, helc, P1, P2, F1, F2, dsig_prefactor,
                                 ReH, ImH, ReHt, ImHt, ReE, ImE)
        
        d_sigma = T_BH_2+I+T_DVCS_2

        return d_sigma

    def compute_BH_amplitude(self, Q2, t, x, phi, e2, y, K, P1, P2, F1, F2, dsig_prefactor):
        """
        Computes the Bethe-Heitler amplitude squared term in nb using inputs from compute_kinematics
        """

        c0_BH, c1_BH, c2_BH, s1_BH = self.BH_UP_coeffs(
            Q2, t, x, e2, y, K, F1, F2)

        # BKM 2002 : Eq (25)
        TBH2 = 1./(x*x*y*y*(1.+e2)*(1.+e2)*t*P1*P2) * (c0_BH + c1_BH*torch.cos(phi) + c2_BH *
                                                       torch.cos(2.*phi) + s1_BH*torch.sin(phi))

        TBH2 *= self.GeV_to_nb*dsig_prefactor

        return TBH2
    
    def compute_Interference(self, Q2, t, x, phi, e2, y, K, t_min, K_tilde, helc, P1, P2, F1, F2, dsig_prefactor,
                             ReH, ImH, ReHt, ImHt, ReE, ImE):
        """
        Computes the Interference term amplitude squared term in nb using inputs from compute_kinematics
        """

        c0_I, c1_I, c2_I, c3_I, s1_I, s2_I = self.I_UP_coeffs(Q2, t, x, e2, y, K, t_min, K_tilde, helc, F1, F2,
                                                              ReH, ImH, ReHt, ImHt, ReE, ImE)

        I = 1./(x*y*y*y*t*P1*P2) * \
            (c0_I + c1_I*torch.cos(phi) + c2_I*torch.cos(2.*phi) +
             c3_I*torch.cos(3.*phi) + s1_I*torch.sin(phi) + s2_I*torch.sin(2.*phi))

        I *= self.GeV_to_nb*dsig_prefactor

        return I

    def form_factors(self, t):
        """
        Computes the Dirac (F1) and Pauli (F2) proton form factors using the dipole approximation for Sachs form factors.
        Based on the dipole ansatz used in the UVA Spin Physics DVCS code.

        Parameters:
            t (float): Momentum transfer squared (GeV^2)

        Returns:
            tuple: (F1, F2)
                F1 (float): Dirac form factor
                F2 (float): Pauli form factor
        """

        GM0 = 2.792847337  # Proton magnetic moment
        GE = 1./((1.-t/0.710649)*(1.-t/0.710649))  # Sachs electric FF (dipole)
        GM = GM0 * GE  # Sachs magnetic FF (dipole)
        F2 = (GM-GE)/(1.-(t/(4.*self.M2)))  # Pauli FF
        F1 = GM - F2  # Dirac FF

        return F1, F2
    
    def BH_UP_coeffs(self, Q2, t, x, e2, y, K, F1, F2):
        """
        Computes the Fourier coefficients (c0, c1, c2, s1) of the unpolarized target Bethe-Heitler squared amplitude. 
        Based on Eq. (35-37) of the BKM 2002 paper.

        Parameters:
            Q2 (float): Photon virtuality (GeV^2)
            t (float): Momentum transfer squared (GeV^2)
            x (float): Bjorken scaling variable
            e2 (float): ε², ratio of longitudinal to transverse photon flux, BKM 2010 - Eq (1.2)
            y (float): Lepton energy loss fraction
            K (float): Kinematic factor, BKM 2010 - below Eq (2.21)
            F1 (float): Dirac form factor
            F2 (float): Pauli form factor

        Returns:
            tuple: (c0, c1, c2, s1) Fourier coefficients
        """

        A = (F1*F1-(t/(4.*self.M2))*F2*F2)
        B = (F1+F2)*(F1+F2)

        c0 = (

            8.*K*K*((2.+3.*e2)*(Q2/t)*A + 2.*x*x*B) +

            (2.-y)*(2.-y) * (
                (2.+e2)*((4.*x*x*self.M2/t)*(1.+t/Q2)*(1.+t/Q2) + 4.*(1.-x)*(1.+x*t/Q2))*A +
                4.*x*x*(x+(1.-x+0.5*e2)*(1.-t/Q2) *
                        (1.-t/Q2)-x*(1.-2.*x)*(t*t/(Q2*Q2)))*B
            ) +

            8.*(1.+e2)*(1.-y-e2*y*y/4.) *
            (2.*e2*(1.-t/(4.*self.M2))*A-x*x*(1.-t/Q2)*(1.-t/Q2)*B)

        )

        c1 = 8.*K*(2.-y)*((4.*x*x*self.M2/t-2.*x-e2)*A + 2.*x*x*(1.-(1.-2.*x)*t/Q2)*B)

        c2 = 8.*x*x*K*K*((4.*self.M2/t)*A + 2.*B)

        s1 = 0.

        return c0, c1, c2, s1
    
    def I_UP_coeffs(self, Q2, t, x, e2, y, K, t_min, K_tilde, helc, F1, F2, ReH, ImH, ReHt, ImHt, ReE, ImE):
        """
        Computes Fourier coefficients for the interference term in the unpolarized Bethe-Heitler + DVCS cross section, 
        based on Eqn (2.35) BKM 2010 paper.

        Parameters:
            Q2 (float): Photon virtuality (GeV^2)
            t (float): Momentum transfer squared (GeV^2)
            x (float): Bjorken scaling variable
            e2 (float): ε², ratio of longitudinal to transverse photon flux, BKM 2010 - Eq (1.2)
            y (float): Lepton energy loss fraction
            K (float): Kinematic factor, BKM 2010 - below Eq (2.21)
            t_min (float): Minimum kinematically allowed value of t, BKM 2002 - Eq (31)
            K_tilde (float): Modified kinematic factor, BKM 2010 - Eq (2.5)
            helc (float): Lepton polarisation (can be 1.0, 0.0 or -1.0)
            F1 (float): Dirac form factor
            F2 (float): Pauli form factor
            ReH (float): Real part of H CFF
            ImH (float): Imaginary part of H CFF
            ReHt (float): Real part of H~ CFF
            ImHt (float): Imaginary part of H~ CFF
            ReE (float): Real part of E CFF
            ImE (float): Real part of E CFF

        Returns:
            tuple: (c0, c1, c2, c3, s1, s2) 
            Returns total Interference term Fourier coefficients
        """

        # One would need 𝓕_eff and 𝓕_T to be able to include the helicity changing terms. Skip for now.

        A = (x/(2.-x+x*t/Q2))

        # BKM 2010 : Eq (2.28)
        ReCC = F1*ReH-(0.25*t/self.M2)*F2*ReE+A*(F1+F2)*ReHt
        ImCC = F1*ImH-(0.25*t/self.M2)*F2*ImE+A*(F1+F2)*ImHt

        # BKM 2010 : Eq (2.29)
        ReCC_V = A*(F1+F2)*(ReH+ReE)
        ImCC_V = A*(F1+F2)*(ImH+ImE)

        # BKM 2010 : Eq (2.30)
        ReCC_A = A*(F1+F2)*ReHt
        ImCC_A = A*(F1+F2)*ImHt

        # First term of BKM 2010 : Eq (2.35)
        c0_A, c1_A, c2_A, c3_A, s1_A, s2_A = self.coeffs_plus_plus_UP(Q2, t, x, e2, y, K, t_min, K_tilde, helc,
                                                                 ReCC, ReCC_V, ReCC_A, ImCC, ImCC_V, ImCC_A)

        # Second term of BKM 2010 : Eq (2.35) - Need to pass CFF_eff to coeffs_zero_plus_UP. Skip for now.
        c0_B, c1_B, c2_B, c3_B, s1_B, s2_B = 0., 0., 0., 0., 0., 0.

        # Third term of BKM 2010 : Eq (2.35) - Need to pass CFF_T to coeffs_minus_plus_UP. Skip for now.
        c0_C, c1_C, c2_C, c3_C, s1_C, s2_C = 0., 0., 0., 0., 0., 0.

        c0 = c0_A + c0_B + c0_C
        c1 = c1_A + c1_B + c1_C
        c2 = c2_A + c2_B + c2_C
        c3 = c3_A + c3_B + c3_C
        s1 = s1_A + s1_B + s1_C
        s2 = s2_A + s2_B + s2_C

        return c0, c1, c2, c3, s1, s2

    @staticmethod
    def coeffs_plus_plus_UP(Q2, t, x, e2, y, K, t_min, K_tilde, helc,
                            ReCC, ReCC_V, ReCC_A, ImCC, ImCC_V, ImCC_A):
        """
        Computes helicity-conserving unpolarised target coefficients. Based on BKM 2010 - Eq (A1), Eq (2.35)
        These are all the plus-plus components.

        Parameters:
            Q2 (float): Photon virtuality (GeV^2)
            t (float): Momentum transfer squared (GeV^2)
            x (float): Bjorken scaling variable
            e2 (float): ε², ratio of longitudinal to transverse photon flux, BKM 2010 - Eq (1.2)
            y (float): Lepton energy loss fraction
            K (float): Kinematic factor, BKM 2010 - below Eq (2.21)
            t_min (float): Minimum kinematically allowed value of t, BKM 2002 - Eq (31)
            K_tilde (float): Modified kinematic factor, BKM 2010 - Eq (2.5)
            helc (float): Lepton polarisation (can be 1.0, 0.0 or -1.0)
            ReCC (float): Real CFF combinations, BKM 2010 : Eq (2.28)
            ReCC_V (float): Real CFF combinations, BKM 2010 : Eq (2.29)
            ReCC_A (float): Real CFF combinations, BKM 2010 : Eq (2.30)
            ImCC (float): Imaginary CFF combinations, BKM 2010 : Eq (2.28)
            ImCC_V (float): Imaginary CFF combinations, BKM 2010 : Eq (2.29)
            ImCC_A (float): Imaginary CFF combinations, BKM 2010 : Eq (2.30)

        Returns:
            tuple: (c0, c1, c2, c3, s1, s2)
            Returns plus-plus Fourier coefficients
        """

        EE = (1.+e2)*(1.+e2)
        RE = torch.sqrt(1.+e2)
        RE5 = torch.pow(RE, 5.)
        TQ = t/Q2
        KTQ = K_tilde*K_tilde/Q2
        A = (1.-y-e2*y*y/4.)
        tp = t-t_min
        TPQ = tp/Q2

        c0_pp = -(4.*(2.-y)*(1.+RE)/EE) * \
            (KTQ*(2.-y)*(2.-y)/RE + TQ*A*(2.-x) *
            (1.+(2.*x*(2.-x+0.5*(RE-1.)+0.5*e2/x)*TQ+e2)/((2.-x)*(1.+RE))))

        c0_pp_V = (8.*(2.-y)*x*TQ/EE) * \
            ((2.-y)*(2.-y)*(KTQ/RE) + A*0.5*(1.+RE)
            * (1.+TQ)*(1.+(RE-1.+2.*x)*TQ/(1.+RE)))

        c0_pp_A = (8.*(2.-y)*TQ/EE) * \
            (KTQ*(2.-y)*(2.-y)*0.5*(1.+RE-2.*x)/RE +
            A*(0.5*(1.+RE) * (1.+RE-x+(RE-1.+x*(3.+RE-2.*x)/(1.+RE))*TQ) - 2.*KTQ))

        c1_pp = -(16.*K*A/RE5)*((1.+(1.-x)*0.5*(RE-1.)/x+0.25*e2/x)*x*TQ-0.25*3.*e2) \
            - 4.*K*(2.-2.*y+y*y+0.5*e2*y*y)*((1.+RE-e2)/RE5) * \
            (1.-(1.-3.*x)*TQ + (1.-RE+3.*e2)*x*TQ/(1.+RE-e2))

        c1_pp_V = (16.*K*x*TQ/RE5) * \
            ((2.-y)*(2.-y)*(1.-(1.-2.*x)*TQ)+A*(1.+RE-2.*x)*TPQ*0.5)

        c1_pp_A = -(16.*K*TQ/EE) * \
            (A*(1.-(1.-2.*x)*TQ+(4.*x*(1.-x)+e2)*TPQ/(4.*RE)) - (2.-y)*(2.-y) *
            (1.-0.5*x+0.25*(1.+RE-2.*x)*(1.-TQ) + (4.*x*(1.-x)+e2)*TPQ/(2.*RE)))

        c2_pp = (8.*(2.-y)*A/EE) * \
            (2.*e2*KTQ/(RE*(1.+RE)) + x*TQ*TPQ*(1.-x-0.5*(RE-1.)+0.5*e2/x))

        c2_pp_V = (8.*(2.-y)*A/EE)*x*TQ * \
            (4.*KTQ/RE + 0.5*(1.+RE-2.*x)*(1.+TQ)*TPQ)

        c2_pp_A = (4.*(2.-y)*A/EE)*TQ * \
            (4.*(1.-2.*x)*KTQ/RE - (3.-RE-2.*x+e2/x)*x*TPQ)

        c3_pp = -(8.*K*A*(RE-1.)/RE5)*((1.-x)*TQ+0.5*(RE-1.)*(1.+TQ))

        c3_pp_V = -(8.*K*A*x*TQ/RE5)*(RE-1.+(1.+RE-2.*x)*TQ)

        c3_pp_A = (16.*K*A*TQ*TPQ/RE5)*(x*(1.-x)+0.25*e2)

        s1_pp = helc * (8.*K*(2.-y)*y/(1.+e2)) * (1.+(1.-x+0.5*(RE-1))*TPQ/(1.+e2))

        s1_pp_V = -helc * (8.*K*(2.-y)*y*x*TQ/EE) * (RE-1.+(1.+RE-2.*x)*TQ)

        s1_pp_A = helc * (8.*K*(2.-y)*y*TQ/(1.+e2)) * \
            (1.-(1.-2.*x)*(1.+RE-2.*x)*TPQ*0.5/RE)

        s2_pp = -helc * (4.*A*y*(1.+RE-2.*x)*TPQ/(torch.pow(RE, 3.))) * \
            ((e2-x*(RE-1.))/(1.+RE-2.*x) - (2.*x+e2)*TPQ/(2.*RE))

        s2_pp_V = -helc * (4.*A*y*x*TQ*(1.-(1.-2.*x)*TQ)/EE) * \
            (RE-1.+(1.+RE-2.*x)*TQ)

        s2_pp_A = -helc * (8.*A*y*TQ*TPQ*(1.+RE-2.*x)/EE) * \
            (1.+(4.*(1.-x)*x+e2)*TQ/(4.-2.*x+3.*e2))

        # BKM 2010 : Eq (2.36)

        CC0_pp = ReCC + c0_pp_V*ReCC_V/c0_pp + c0_pp_A*ReCC_A/c0_pp
        CC1_pp = ReCC + c1_pp_V*ReCC_V/c1_pp + c1_pp_A*ReCC_A/c1_pp
        CC2_pp = ReCC + c2_pp_V*ReCC_V/c2_pp + c2_pp_A*ReCC_A/c2_pp
        CC3_pp = ReCC + c3_pp_V*ReCC_V/c3_pp + c3_pp_A*ReCC_A/c3_pp

        # compute SS only when denominator is non-zero, else output ImCC
        
        SS1_pp = torch.where(s1_pp != 0.0,
                        ImCC + s1_pp_V*ImCC_V/s1_pp + s1_pp_A*ImCC_A/s1_pp,
                        ImCC)
        SS2_pp = torch.where(s2_pp != 0.0,
                        ImCC + s2_pp_V*ImCC_V/s2_pp + s2_pp_A*ImCC_A/s2_pp,
                        ImCC)

        # First term of BKM 2010 : Eq (2.35)
        
        c0 = c0_pp*CC0_pp
        c1 = c1_pp*CC1_pp
        c2 = c2_pp*CC2_pp
        c3 = c3_pp*CC3_pp
        s1 = s1_pp*SS1_pp
        s2 = s2_pp*SS2_pp

        return c0, c1, c2, c3, s1, s2

    @staticmethod
    def coeffs_zero_plus_UP(Q2, t, x, e2, y, K, t_min, K_tilde, helc,
                            ReCC_eff, ReCC_V_eff, ReCC_A_eff, ImCC_eff, ImCC_V_eff, ImCC_A_eff):
        """
        Computes longitudinal-transverse unpolarised target helicity-changing coefficients. Based on BKM 2010 - Eq (A2-A3), Eq (2.35) 
        These are all the 0-plus components.

        Parameters:
            Q2 (float): Photon virtuality (GeV^2)
            t (float): Momentum transfer squared (GeV^2)
            x (float): Bjorken scaling variable
            e2 (float): ε², ratio of longitudinal to transverse photon flux, BKM 2010 - Eq (1.2)
            y (float): Lepton energy loss fraction
            K (float): Kinematic factor, BKM 2010 - below Eq (2.21)
            t_min (float): Minimum kinematically allowed value of t, BKM 2002 - Eq (31)
            K_tilde (float): Modified kinematic factor, BKM 2010 - Eq (2.5)
            helc (float): Lepton polarisation (can be 1.0, 0.0 or -1.0)
            ReCC_eff (float): Real CFF_effective combinations, BKM 2010 : Eq (2.28)
            ReCC_V_eff (float): Real CFF_effective combinations, BKM 2010 : Eq (2.29)
            ReCC_A_eff (float): Real CFF_effective combinations, BKM 2010 : Eq (2.30)
            ImCC_eff (float): Imaginary CFF_effective combinations, BKM 2010 : Eq (2.28)
            ImCC_V_eff (float): Imaginary CFF_effective combinations, BKM 2010 : Eq (2.29)
            ImCC_A_eff (float): Imaginary CFF_effective combinations, BKM 2010 : Eq (2.30)

        Returns:
            tuple: (c0, c1, c2, c3, s1, s2)
            Returns 0-plus Fourier coefficients
        """

        EE = (1.+e2)*(1.+e2)
        RE = torch.sqrt(1.+e2)
        RE5 = torch.pow(RE, 5.)
        TQ = t/Q2
        KTQ = K_tilde*K_tilde/Q2
        A = (1.-y-e2*y*y/4.)
        tp = t-t_min
        TPQ = tp/Q2

        c0_0p = (12.*torch.sqrt(2.*A)*K*(2.-y)/RE5) * (e2+(2.-6.*x-e2)*TQ/3.)

        c1_0p = (8.*torch.sqrt(2.*A)/EE) * \
            ((2.-y)*(2.-y)*TPQ*(1.-x+((1.-x)*x+0.25*e2)*TPQ/RE) +
            (A/RE)*(1.-(1.-2.*x)*TQ)*(e2-2.*(1.+0.5*e2/x)*x*TQ))

        c2_0p = -(8.*torch.sqrt(2.*A)*K*(2.-y)*(1.+0.5*e2)/RE5) * \
            (1. + (1. + 0.5*e2/x)*x*TQ/(1.+0.5*e2))

        s1_0p = helc*8.*torch.sqrt(2.*A)*(2.-y)*y*KTQ/EE

        s2_0p = helc*(8.*torch.sqrt(2.*A)*K*y*(1.+0.5*e2)/EE) * \
            (1. + (1.+0.5*e2/x)*x*TQ/(1.+0.5*e2))

        c0_0p_V = (24.*torch.sqrt(2.*A)*K*(2.-y)*x*TQ/RE5)*(1.-(1.-2.*x)*TQ)

        c1_0p_V = (16.*torch.sqrt(2.*A)*x*TQ/RE5) * \
            (KTQ*(2.-y)*(2.-y) + (1.-(1.-2.*x)*TQ)*(1.-(1.-2.*x)*TQ)*A)

        c2_0p_V = c0_0p_V/3.

        s1_0p_V = helc*(4.*torch.sqrt(2.*A)*y*(2.-y)*x*TQ/EE) * \
            (4.*(1.-2.*x)*TQ*(1.+x*TQ)+e2*(1.+TQ)*(1.+TQ))

        s2_0p_V = -helc*(8.*torch.sqrt(2.*A)*K*y*x*TQ/EE)*(1.-(1.-2.*x)*TQ)

        c0_0p_A = (4.*torch.sqrt(2.*A)*K*(2.-y)*TQ*(8.-6.*x+5.*e2)/RE5) * \
            (1.-TQ*(2.-12.*x*(1.-x)-e2)/(8.-6.*x+5.*e2))

        c1_0p_A = (8.*torch.sqrt(2.*A)*TQ/RE5) * \
            (KTQ*(1.-2.*x)*(2.-y)*(2.-y)+(1.-(1.-2.*x)*TQ)
            * A*(4.-2.*x+3.*e2+TQ*(4.*x*(1.-x)+e2)))

        c2_0p_A = (8.*torch.sqrt(2.*A)*K*(2.-y)*TQ/EE) * \
            (1.-x+0.5*TPQ*(4.*x*(1.-x)+e2)/RE)

        s1_0p_A = -helc*8.*torch.sqrt(2.*A)*y*(2.-y)*(1.-2.*x)*TQ*K*K/EE

        s2_0p_A = -helc*(2.*torch.sqrt(2.*A)*K*y*TQ/EE) * \
            (4.-4.*x+2.*e2+4.*TQ*(4.*x*(1.-x)+e2))

        # BKM 2010 : Eq (2.37)

        B = torch.sqrt(2)*KTQ/(2.-x)

        CC0_0p = B*(ReCC_eff + c0_0p_V*ReCC_V_eff/c0_0p + c0_0p_A*ReCC_A_eff/c0_0p)
        CC1_0p = B*(ReCC_eff + c1_0p_V*ReCC_V_eff/c1_0p + c1_0p_A*ReCC_A_eff/c1_0p)
        CC2_0p = B*(ReCC_eff + c2_0p_V*ReCC_V_eff/c2_0p + c2_0p_A*ReCC_A_eff/c2_0p)

        # compute SS only when denominator is non-zero, else output ImCC

        SS1_0p = torch.where(s1_0p != 0.0,
                        B*ImCC_eff + s1_0p_V*ImCC_V_eff/s1_0p + s1_0p_A*ImCC_A_eff/s1_0p,
                        B*ImCC_eff)
        SS2_0p = torch.where(s2_0p != 0.0,
                        B*ImCC_eff + s2_0p_V*ImCC_V_eff/s2_0p + s2_0p_A*ImCC_A_eff/s2_0p,
                        B*ImCC_eff)

        # Second term of BKM 2010 : Eq (2.35)

        c0 = c0_0p*CC0_0p
        c1 = c1_0p*CC1_0p
        c2 = c2_0p*CC2_0p
        c3 = 0.
        s1 = s1_0p*SS1_0p
        s2 = s2_0p*SS2_0p

        return c0, c1, c2, c3, s1, s2

    @staticmethod
    def coeffs_minus_plus_UP(Q2, t, x, e2, y, K, t_min, K_tilde, helc,
                            ReCC_T, ReCC_V_T, ReCC_A_T, ImCC_T, ImCC_V_T, ImCC_A_T):
        """
        Computes transverse-transverse unpolarised target helicity-changing coefficients. Based on BKM 2010 - Eq (A4), Eq (2.35) 
        These are all the minus-plus components.

        Parameters:
            Q2 (float): Photon virtuality (GeV^2)
            t (float): Momentum transfer squared (GeV^2)
            x (float): Bjorken scaling variable
            e2 (float): ε², ratio of longitudinal to transverse photon flux, BKM 2010 - Eq (1.2)
            y (float): Lepton energy loss fraction
            K (float): Kinematic factor, BKM 2010 - below Eq (2.21)
            t_min (float): Minimum kinematically allowed value of t, BKM 2002 - Eq (31)
            K_tilde (float): Modified kinematic factor, BKM 2010 - Eq (2.5)
            helc (float): Lepton polarisation (can be 1.0, 0.0 or -1.0)
            ReCC_T (float): Real CFF_T combinations, BKM 2010 : Eq (2.28)
            ReCC_V_T (float): Real CFF_T combinations, BKM 2010 : Eq (2.29)
            ReCC_A_T (float): Real CFF_T combinations, BKM 2010 : Eq (2.30)
            ImCC_T (float): Imaginary CFF_T combinations, BKM 2010 : Eq (2.28)
            ImCC_V_T (float): Imaginary CFF_T combinations, BKM 2010 : Eq (2.29)
            ImCC_A_T (float): Imaginary CFF_T combinations, BKM 2010 : Eq (2.30)

        Returns:
            tuple: (c0, c1, c2, c3, s1, s2)
            Returns minus-plus Fourier coefficients
        """

        EE = (1.+e2)*(1.+e2)
        RE = torch.sqrt(1.+e2)
        RE5 = torch.pow(RE, 5.)
        TQ = t/Q2
        KTQ = K_tilde*K_tilde/Q2
        A = (1.-y-e2*y*y/4.)
        tp = t-t_min
        TPQ = tp/Q2

        c0_mp = (8.*(2.-y)/(torch.pow(RE, 3.))) * \
            ((2.-y)*(2.-y)*(RE-1.)*KTQ*0.5/(1.+e2) +
            A*(1.-x-0.5*(RE-1.)+0.5*e2/x)*x*TQ*TPQ/RE)

        c1_mp = ((8.*K/torch.pow(RE, 3.)))*((2.-y)*(2.-y)*((2.-RE)/(1.+e2)) *
                                        ((RE-1.+e2)*(1.-TQ)*0.5/(2.-RE)-x*TQ) +
                                        (2.*A/RE)*((1.-RE+0.5*e2)*0.5/RE+TQ*(1.-3.*0.5*x+(x+0.5*e2)*0.5/RE)))

        c2_mp = (4.*(2.-y)*A*(1.+RE)/RE5) * \
            ((2.-3.*x)*TQ+(1.-2.*x+(2.*(1.-x))/(1.+RE)) *
            x*TQ/Q2 + (1.+(RE+x+(1.-x)*TQ)*TQ/(1.+RE))*e2)

        c3_mp = -(8.*K*A*(1.+RE+0.5*e2)/RE5) * \
            (1. + (1.+RE+0.5*e2/x)*x*TQ/(1.+RE+0.5*e2))

        s1_mp = helc*(4.*K*(2.-y)*y/EE) * (1.-RE+2.*e2-2.*(1.+0.5*(RE-1.)/x)*x*TQ)

        s2_mp = helc*(2.*y*A*(1.+RE)/EE)*(e2-2.*(1.+0.5*e2/x)*x*TQ) * \
            (1.+(RE-1.+2.*x)*TQ/(1.+RE))

        c0_mp_V = (4.*(2.-y)*x*TQ/RE5)*(2.*KTQ*(2.-2.*y+y*y+0.5*y*y*e2)-(1.-(1.-2.*x)*TQ)*A) * \
            (RE-1.+(RE+1.-2.*x)*TQ)

        c1_mp_V = (8.*K*x*TQ/RE5)*(2.*(1.-(1.-2.*x)*TQ)*(2.-2.*y+y*y+0.5*y*y*e2) +
                                A*(3.-RE-(3.*(1.-2.*x)+RE)*TQ))

        c2_mp_V = (4.*(2.-y)*A*x*TQ/RE5) * \
            (4.*KTQ+1.+RE+TQ*((1.-2.*x)*(1.-2.*x-RE)*TQ-2.+4.*x+2.*x*RE))

        c3_mp_V = (8.*K*A*x*TQ*(1.+RE)/RE5)*(1.-TQ*(1.-2.*x-RE)/(1.+RE))

        s1_mp_V = helc*(8.*K*y*(2.-y)*x*TQ*(1.+RE)/EE) * \
            (1.-TQ*(1.-2.*x-RE)/(1.+RE))

        s2_mp_V = helc*(4.*y*A*x*TQ*(1.+RE)/EE) * \
            (1.-(1.-2.*x)*TQ) * (1.-TQ*(1.-2.*x-RE)/(1.+RE))

        c0_mp_A = (4.*(2.-y)*TQ/EE)*(TPQ*A*(2.*x*x-e2-3.*x+x*RE) +
                                    (KTQ/RE)*(4.-2.*x*(2.-y)*(2.-y)-4.*y+y*y-torch.pow(RE, 3.)))

        c1_mp_A = (4.*K*TQ/RE5)*((2.-2.*y+y*y+0.5*y*y*e2)*(5.-4.*x+3.*e2-RE-TQ*(1.-e2-RE-2.*x*(4.-4.*x-RE))) +
                                A*(8.+5.*e2-6.*x+2.*x*RE-TQ*(2.-e2+2.*RE-4.*x*(3.-3.*x+RE))))

        c2_mp_A = (16.*(2.-y)*A*TQ/torch.pow(RE, 3))*(KTQ*(1.-2.*x)/(1.+e2)-(1.-x)*(2.*x*x-e2-3.*x-x*RE)/(4.*x*(1.-x)+e2) -
                                                TPQ*(2.*x*x-e2-3.*x-x*RE)*0.25/RE)

        c3_mp_A = (16.*K*A*TQ/EE)*(1.-x+TPQ*(4.*x*(1.-x)+e2)*0.25/RE)

        s1_mp_A = helc*(4.*K*y*(2.-y)*TQ/EE) * \
            (3.+2.*e2+RE-2.*x-2.*x*RE-TQ*(1.-2.*x)*(1.-2.*x-RE))

        s2_mp_A = helc*(2.*A*TQ/EE)*(4.-2.*x+3.*e2+TQ*(4.*x*(1.-x)+e2)) * \
            (1.+RE-TQ*(1.-2.*x-RE))

        # BKM 2010 : Eq (2.37) Maybe??

        B = torch.sqrt(2)*KTQ/(2.-x)

        CC0_mp = B*(ReCC_T + c0_mp_V*ReCC_V_T/c0_mp + c0_mp_A*ReCC_A_T/c0_mp)
        CC1_mp = B*(ReCC_T + c1_mp_V*ReCC_V_T/c1_mp + c1_mp_A*ReCC_A_T/c1_mp)
        CC2_mp = B*(ReCC_T + c2_mp_V*ReCC_V_T/c2_mp + c2_mp_A*ReCC_A_T/c2_mp)
        CC3_mp = B*(ReCC_T + c3_mp_V*ReCC_V_T/c3_mp + c3_mp_A*ReCC_A_T/c3_mp)

        # compute SS only when denominator is non-zero, else output ImCC

        SS1_mp = torch.where(s1_mp != 0.0,
                        B*ImCC_T + s1_mp_V*ImCC_V_T/s1_mp + s1_mp_A*ImCC_A_T/s1_mp,
                        B*ImCC_T)
        SS2_mp = torch.where(s2_mp != 0.0,
                        B*ImCC_T + s2_mp_V*ImCC_V_T/s2_mp + s2_mp_A*ImCC_A_T/s2_mp,
                        B*ImCC_T)

        # Last term of BKM 2010 : Eq (2.35)

        c0 = c0_mp*CC0_mp
        c1 = c1_mp*CC1_mp
        c2 = c2_mp*CC2_mp
        c3 = c3_mp*CC3_mp
        s1 = s1_mp*SS1_mp
        s2 = s2_mp*SS2_mp

        return c0, c1, c2, c3, s1, s2