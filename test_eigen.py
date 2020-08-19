import numpy as np 

import openmdao.api as om 

class EigenVal2(om.ImplicitComponent): 

    def setup(self): 

        self.add_input('A', shape=(2,2))

        self.add_output('lambda', shape=2)

        self.declare_partials('*', '*', method='cs')

    def apply_nonlinear(self, inputs, outputs, residuals): 

        l0 = outputs['lambda'][0]
        l1 = outputs['lambda'][1]

        A = inputs['A']
        a,b,c,d = A[0,0], A[0,1], A[1,0], A[1,1]
        residuals['lambda'][0] = l0**2 - (a+d)*l0 + (a*d-b*c)
        residuals['lambda'][1] = l1**2 - (a+d)*l1 + (a*d-b*c)

    # def compute_partials(self, inputs, outputs, J): 

    #     pass


    def solve_nonlinear(self, inputs, outputs): 

        #(A-lambda I) ... characteristic equation 
        # (a-lamba)(d-lambda) - bc
        # lambda**2 - (a+d)*lambda + (ad-bc) = 0 

        A = inputs['A']
        a,b,c,d = A[0,0], A[0,1], A[1,0], A[1,1]
        coeffs = [1, -(a+d), (a*d-b*c)]

        outputs['lambda'] = np.roots(coeffs)


if __name__ == "__main__": 

    p = om.Problem()

    p.model.add_subsystem('eigen', EigenVal2())

    # optional
    newton = p.model.nonlinear_solver = om.NewtonSolver()
    newton.options['solve_subsystems'] = True
    newton.options['iprint'] = 2

    p.model.linear_solver = om.DirectSolver()

    p.setup()

    p['eigen.A'] = [[1, 2], 
                    [3, 4]]

    p.run_model()

    p.model.list_outputs(print_arrays=True, residuals=True)

    check_eig = np.linalg.eig(p['eigen.A'])[0]

    # print('sanity_check', check_eig)


    J = p.compute_totals(of=['eigen.lambda'], wrt=['eigen.A'])
    print(J)