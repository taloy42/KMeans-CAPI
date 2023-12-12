"""
contains all the invoke commands
"""
from invoke import task
import sys

@task
def run(c, k='sup', n='soup', Random=True):
    K_max2 = 20
    n_max2 = 500

    K_max3 = 20
    n_max3 = 500
    if (not (k.isdigit() and n.isdigit())) and not Random:
        print("Parameters not sufficient, input should be:\n\
	n: 0 < integer\n\
	k: 0 < integer < n\n\
	flag of randomization: -R, --Random for true, --no-Random for false")
        exit()
    build(c)
    print("\nMaximum capacity for dimension 2\n\
Maximum n = %d\tMaximum K = %d\n\
Maximum capacity for dimension 3\n\
Maximum n = %d\tMaximum K = %d\n\n" % (n_max2, K_max2, n_max3, K_max3))
    if Random:
    	c.run(r"python3.8.5 main.py 100 5 --rand")
    else:
        c.run(r"python3.8.5 main.py {} {}".format(n,k))


@task
def build(c):
    c.run(r"python3.8.5 setup.py build_ext --inplace")


@task(aliases=['del'])
def delete(c):
    c.run(r"rm *mykmeanssp*.so")


if __name__ == '__main__':
    print(sys.version_info)
