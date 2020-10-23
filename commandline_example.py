#demonstrates running from command line with flags

#recommended command line module in python
import argparse

#command line
parser=argparse.ArgumentParser(description="tutorial")

#explicit Arguments
#positional arguments
parser.add_argument("double_me", help="prints the doubled input", type=float)
#flag accepts argument
parser.add_argument("--print_me", help="prints the input")
#flag accepts no argument
parser.add_argument("--double_hello", help="doubles printing of Hello World!", action="store_true")

args = parser.parse_args()

print(args.double_me*2)

if args.print_me:
    print(args.print_me)

if args.double_hello:
    print("Hello World")
    print("Hello World")
