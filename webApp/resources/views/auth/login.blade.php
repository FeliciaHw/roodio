<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>ROODIO - Login</title>
    @vite(['resources/css/app.css', 'resources/js/app.js'])
    <script type="text/javascript" src="{{ asset('js/bg-effect.js') }}" defer></script>
    <script type="text/javascript" src="{{ asset('js/particle-network.js') }}" defer></script>
</head>
<body>
    <div class='grid w-screen h-screen bg-primary-100 overflow-hidden justify-items-center items-center mx-auto'>
        <div id="particle-canvas"></div>
        <div class='w-xs h-[26rem] border-secondary-happy-50 border-3 bg-white/75 hover:bg-white/85 transition-colors active:bg-white/85 ease-linear duration-150 rounded-4xl col-start-1 row-start-1 z-10 flex flex-col items-center justify-around shadow-xl shadow-secondary-happy-50/40 md:w-md md:h-[32rem]'>
            <div class='w-[inherit] animate-pulse mt-4 flex flex-row-max h-max items-center justify-center font-primary text-title font-bold text-primary-85 md:text-hero'>
                <span>L</span>
                <div class='rounded-full w-12 h-12 bg-primary-85 mx-2 flex justify-center items-center md:w-16 md:h-16'>
                    <img src="{{ asset('assets/logo-no-text.png') }}" alt="logo" class='w-5 animate-spin bg-amber-200 rounded-full drop-shadow-md/70 drop-shadow-white/70 md:w-7'>
                </div>
                <span class='tracking-[.5rem]'>GI</span>
                <span>N</span>
            </div>
            <p class='font-secondaryAndButton -mt-3 text-small text-primary-50 md:text-body md:-mt-9 md:tracking-[.05rem]'>Welcome back to ROODIO!</p>
            <div class='font-secondaryAndButton text-body flex justify-center w-75 h-max'>
                <form action="" method='POST'>
                    <div class='flex flex-col w-3xs mb-8 md:w-sm md:mb-11'>
                        <label for="username" class='mb-1 text-primary-70 md:mb-1.5'>
                            <img src="{{ asset('assets/icon/user.svg') }}" alt="user" class='w-5 inline'>
                            <p class='align-middle inline'>Username</p>
                        </label>
                        <input type="text" name="username" id="username" placeholder="Input your username..." class='not-placeholder-shown:bg-accent-20/60 text-small outline-none border-b rounded-md px-1.5 py-0.5 border-shadedOfGray-50 placeholder:text-micro focus:border-amber-600 focus:border-b-2 focus:bg-secondary-happy-20/50 ease-in-out duration-150 hover:bg-shadedOfGray-20/90 md:text-body md:h-8 md:placeholder:text-small'>
                    </div>
                    <div class='flex flex-col w-3xs relative mb-10 md:w-sm md:mb-11'>
                        <label for="password" class='mb-1 text-primary-70 relative'>
                            <img src="{{ asset('assets/icon/password.svg') }}" alt="password" class='w-5 inline'>
                            <p class='align-middle inline'>
                                Password
                                <a href='/forget-password' class='right-0 top-1/2 -translate-y-1/2 absolute text-xs font-bold text-secondary-sad-100 md:text-micro hover:text-primary-50'>Forget Password?</a>
                            </p>
                            
                        </label>
                        <input type="password" name="password" id="password" placeholder="Input your password..." class='not-placeholder-shown:bg-accent-20/60 text-small outline-none border-b rounded-md px-1.5 py-0.5 border-shadedOfGray-50 placeholder:text-micro focus:border-amber-600 focus:border-b-2 focus:bg-secondary-happy-20/50 ease-in-out duration-150 hover:bg-shadedOfGray-20/90 md:text-body md:h-8 md:placeholder:text-small'>
                        <button type='button' id='showPass' class='w-4 h-4 absolute right-1.5 bottom-1.5 flex items-center justify-center cursor-pointer md:bottom-2 md:right-2'>
                            <img src="{{ asset('assets/icon/eye-closed.svg') }}" alt="eye-closed">
                            <span class='absolute invisible' id='eye-open'>&#128065;</span>
                        </button>
                    </div>
                    <div class='mb-4'>
                        <button type="submit" class='text-smallBtn font-bold w-3xs font-secondaryAndButton bg-primary-10 text-primary-100 rounded-2xl py-1 mb-2 cursor-pointer hover:bg-primary-50 hover:text-white ease-in-out duration-150 md:w-sm md:h-11'>Login</button>
                        <p class='text-micro text-center md:text-small'>Don't have account? <a href="/sign-up" class='font-bold text-secondary-sad-100 hover:text-primary-50'>Sign Up Here!</a></p>
                    </div>
                </form>
            </div>
        </div>
    </div>
</body>
</html>