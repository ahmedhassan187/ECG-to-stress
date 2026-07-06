"""Replace main.py with refactored version"""
pass

"""Generate the new refactored main.py"""
import os

content = r'''
def main():
    """Main entry point for CLI application."""
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    if args.correlation:
        run_correlation_analysis(args)
    elif args.full_signal:
        run_full_signal_visualization(args)
    elif args.ml_training:
        run_ml_training(args)
    elif args.fft_analysis:
        run_fft_analysis(args)
    elif args.prediction:
        run_prediction(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
'''

dest = r'g:\Master\Thesis\FLT\Code\ECG-to-stress\src\main.py'
with open(dest, 'w', newline='') as f:
    f.write(content)
print("Written")
