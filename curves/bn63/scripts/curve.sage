class BN(object):
    @staticmethod
    def generate_prime_order(zbits):
        while True:
            z = randint(2^(zbits - 1), 2^zbits)
            pz = int(BN.p(z))
            if not is_prime(pz):
                continue
            rz = int(BN.r(z))
            if not is_prime(rz):
                continue
            break
        K = GF(pz)
        print(pz)
        print(rz)
        b = 1
        while True:
            curve = EllipticCurve(K, [0, b])
            card = curve.cardinality()
            if card % rz == 0:
                break
            b += 1
        return curve

    @staticmethod
    def curve(p, r, b):
       F = GF(p)
       R = GF(r)

       p = F.characteristic()
       F2.<G> = GF(p^2)
       F6.<G> = GF(p^6)
       F12.<G> = GF(p^12)

       c = EllipticCurve(F, [0, b]) 

       f = F2.frobenius_endomorphism()
       g = F.multiplicative_generator()
       return f(g).polynomial().coefficients()
       

    @staticmethod
    def p(z):
        return 36 * z^4 + 36 * z^3 + 24 * z^2 + 6 * z + 1
    @staticmethod
    def r(z):
        return 36 * z^4 + 36 * z^3 + 18 * z^2 + 6 * z + 1
    @staticmethod
    def t(z):
        return 6 * z^2 + 1
