<?php
namespace App\Http\Controllers;

use App\Models\Region;
use App\Models\User;
use App\Models\userDetails;
use Carbon\Carbon;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Validation\Rules\Password;

class AuthController extends Controller
{
    public function loginView()
    {
        return view('auth.login');
    }

    public function login(Request $request)
    {
        $credentials = $request->validate([
            'username' => ['required'],
            'password' => ['required'],
        ]);

        if (Auth::attempt($credentials)) {
            $request->session()->regenerate();

            return redirect()->route('welcome');
        }

        return back()->with('failed', 'Failed to login!');
    }

    public function userVerificationView()
    {
        return view('auth.forgetPasswordValidation');
    }

    public function userVerification()
    {
        // the logic is not yet

        return redirect()->route('forgetPassword');
    }

    public function forgetPasswordView()
    {
        return view('auth.forgetPassword');
    }

    public function forgetPassword()
    {
        //the logic is not yet

        return redirect()->route('login');
    }

    public function registerView()
    {
        $regions = Region::orderBy('id')->get();
        return view('auth.register', compact('regions'));
    }

    public function register(Request $request, OtpController $otpController)
    {
        $validated = $request->validate([
            'fullname' => 'required|max:255',
            'email'    => 'required|email|max:255|unique:user_details,email',
            'dob'      => 'required|date',
            'gender'   => 'required|in:1,0,null',
            'country'  => 'required|string|exists:regions,id',
        ]);

        session()->put('register.step1', $validated);

        $otpController->send($validated['email']);

        session(['user_verification_passed' => true]);

        return redirect()->route('register.validation');
    }

    public function registerValidationView()
    {
        return view('auth.registerValidation');
    }

    public function registerValidation(Request $request, OtpController $otpController)
    {
        $otp = collect(range(1, 6))
            ->map(fn($i) => $request->input("otp-$i"))
            ->implode('');

        $request->merge(['otp' => $otp]);

        $request->validate([
            'otp' => 'required|digits:6',
        ], [
            'otp.required' => 'OTP code must be completed.',
            'otp.digits'   => 'OTP code must be 6 digits.',
        ]);

        $session = session('register.step1');
        $verify  = $otpController->verify($session['email'], $request['otp']);
        if ($verify) {
            session(['otp_passed' => true]);
            return redirect()->route('account');
        } else {
            return back()->withErrors(['otp' => 'Invalid or expired OTP']);
        }
    }

    public function accountView()
    {
        $datas     = session('register.step1');
        $firstName = trim(explode(' ', $datas['fullname'])[0]);
        $random    = random_int(100000, 999999);
        $symbols   = ['-', '#', '_', '.', '~'];
        $symbol    = collect($symbols)->random();

        $username = $firstName . $random . $symbol;
        return view('auth.account', compact('username'));
    }

    public function account(Request $request)
    {
        $validated = $request->validate([
            'username'              => 'required|max:25|unique:users,username',
            'password'              => [
                'required',
                'string',
                'confirmed',
                Password::min(8)->letters()->numbers(),
            ],
            'password_confirmation' => 'required',
        ]);

        $user = User::create($validated);

        $datas                = session('register.step1');
        $datas['dateOfBirth'] = $datas['dob'];
        $datas['dateOfBirth'] = Carbon::createFromFormat(
            'm/d/Y',
            $datas['dateOfBirth']
        )->format('Y-m-d');
        unset($datas['dob']);
        $datas['userId'] = $user->id;

        userDetails::create($datas);

        return redirect()->route('login');
    }
}
