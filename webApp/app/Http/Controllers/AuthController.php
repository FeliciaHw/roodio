<?php
namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

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

    public function registerView()
    {
        return view('auth.register');
    }

    public function register()
    {
        
        
        return redirect()->route('otp');
    }

    public function otpView()
    {
        return view('auth.otp');
    }

    public function otp() 
    {
        
        
        return redirect()->route('account');
    }

    public function accountView()
    {
        return view('auth.account');
    }

    public function account()
    {
        

        return redirect()->route('login');
    }
}
