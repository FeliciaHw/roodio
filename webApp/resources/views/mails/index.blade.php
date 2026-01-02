<style>
    @font-face {
        font-family: 'aeonik';
        src: url('../../../resources/fonts/aeonik.ttf');
    }

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        font-family: aeonik;
    }
    .container {
        height: max-content;
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .logo {
        width: 8rem;
        filter: drop-shadow(0 0 3px #FFE9D3);
        margin-bottom: .5rem;
    }

    .header {
        background-color: #06134D;
        width: inherit;
        height: max-content;
        padding: 1rem 0;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .header > p {
        color: #FFD1A6;
        font-size: 2rem;
        font-weight: bold;
        letter-spacing: .1rem;
        text-align: center;
        margin: .5rem 0;
    }

    .content {
        margin: 1.5rem 0;
        width: 90%;
        height: max-content;
        /* background-color: #FFE9D3; */
        font-size: 1rem;
        text-align: justify;
        border: 1px dashed #142C80;
        border-radius: .5rem; 
        padding: .75rem;
    }

    b:not(.dark-bold) {
        color: #FF8E2B;
    }
    
    p {
        line-height: 1.5rem;
    }

    .code-container {
        display: flex;
        justify-content: center;
        height: max-content;
    }

    .code {
        margin: .75rem 0;
        border: .15rem solid #FFC48D;
        border-radius: .5rem;
        font-size: 3rem;
        letter-spacing: .75rem;
        font-weight: bold;
        padding: .25rem 2rem .25rem 2.75rem;
        color: #1F3A98;
    }

    .gray-text {
        color: #808080;
    }

    hr {
        border: .05rem dashed #808080;
        margin-bottom: .5rem;
    }

    .email-image {
        background-color: #A4BEF2;
        width: max-content;
        padding: .3rem;
        border-radius: 100%;
    }

    .email {
        display: flex;
        flex-direction: row;
        align-items: center;
        gap: .5rem;
    }

    .footer {
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
</style>

<div class='container'>
    <div class='header'>
        <img src="{{ asset('assets/logo-with-text.png') }}" alt="logo" class='logo'>
        <p>TWO WAY VERIFICATION</p>
    </div>
    <div class='content'>
        <p>Dear 
            <b class='dark-bold'>
                {{ if ($gender == 1) {
                    echo('Mr.')
                } elseif ($gender == 0) {
                    echo('Ms.')
                }else {
                    echo('')
                } }}
            </b>
            <b>{{ Str::upper($fullname) }}</b>,
        </p>
        <br>
        <br>
        <p>You recently requested a <b>One-Time Password (OTP)</b> to continue with your authentication process. Please use the code below to complete the verification.</p>
        <div class='code-container'>
            <div class='code'>{{ $otp }}</div>
        </div>
        <p>This code is valid <b>for 5 minutes only</b>. For your security, <b>please do not share this OTP with anyone</b>. Our team will never ask for your OTP by email, phone, or any other method.</p>
        <br>
        <p>If you did not request this code, you can safely ignore this email. No changes will be made to your account without verification.</p>
        <br>
        <p>Thank you for using ROODIO. If you need any help or have questions, feel free to contact our support team.</p>
        <br>
        <br>
        <p>Best Regards,</p>
        <p><b class='dark-bold'>ROODIO Team</b></p>
        <br><hr>
        <div class='footer'>
            <p><b>Our Contact</b></p>
            <div class='email'>
                <div class='email-image'>
                    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">

                    <!-- Uploaded to: SVG Repo, www.svgrepo.com, Transformed by: SVG Repo Mixer Tools -->
                    <svg height="1rem" width="1rem" version="1.1" id="_x32_" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 512 512" xml:space="preserve" fill="#000000">

                    <g id="SVGRepo_bgCarrier" stroke-width="0"/>

                    <g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"/>

                    <g id="SVGRepo_iconCarrier"> <style type="text/css"> .st0{fill:#06134D;} </style> <g> <path class="st0" d="M510.678,112.275c-2.308-11.626-7.463-22.265-14.662-31.054c-1.518-1.915-3.104-3.63-4.823-5.345 c-12.755-12.818-30.657-20.814-50.214-20.814H71.021c-19.557,0-37.395,7.996-50.21,20.814c-1.715,1.715-3.301,3.43-4.823,5.345 C8.785,90.009,3.63,100.649,1.386,112.275C0.464,116.762,0,121.399,0,126.087V385.92c0,9.968,2.114,19.55,5.884,28.203 c3.497,8.26,8.653,15.734,14.926,22.001c1.59,1.586,3.169,3.044,4.892,4.494c12.286,10.175,28.145,16.32,45.319,16.32h369.958 c17.18,0,33.108-6.145,45.323-16.384c1.718-1.386,3.305-2.844,4.891-4.43c6.27-6.267,11.425-13.741,14.994-22.001v-0.064 c3.769-8.653,5.812-18.171,5.812-28.138V126.087C512,121.399,511.543,116.762,510.678,112.275z M46.509,101.571 c6.345-6.338,14.866-10.175,24.512-10.175h369.958c9.646,0,18.242,3.837,24.512,10.175c1.122,1.129,2.179,2.387,3.112,3.637 L274.696,274.203c-5.348,4.687-11.954,7.002-18.696,7.002c-6.674,0-13.276-2.315-18.695-7.002L43.472,105.136 C44.33,103.886,45.387,102.7,46.509,101.571z M36.334,385.92V142.735L176.658,265.15L36.405,387.435 C36.334,386.971,36.334,386.449,36.334,385.92z M440.979,420.597H71.021c-6.281,0-12.158-1.651-17.174-4.552l147.978-128.959 l13.815,12.018c11.561,10.046,26.028,15.134,40.36,15.134c14.406,0,28.872-5.088,40.432-15.134l13.808-12.018l147.92,128.959 C453.137,418.946,447.26,420.597,440.979,420.597z M475.666,385.92c0,0.529,0,1.051-0.068,1.515L335.346,265.221L475.666,142.8 V385.92z"/> </g> </g>

                    </svg>
                </div>
                <p>info.roodio@gmail.com</p>
            </div>
        </div>
        <br><hr>
        <p class='gray-text'>This email was sent automatically for security purposes.</p>
        <p><b class='dark-bold gray-text'>Please do not reply to this message.</b></p>

    </div>
</div>