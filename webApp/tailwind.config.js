export default {
  theme: {
    extend: {
      colors: {
        primary: {
          100: '#020A36',
          85: '#06134D',
          70: '#0D1F67',
          60: '#142C80',
          50: '#1F3A98',
          30: '#4F6CC3',
          20: '#7591DB',
          10: '#A4BEF2',
        },

        secondary: {
          happy: {
            100: '#FF8E2B',
            85: '#FFA350',
            70: '#FFB775',
            60: '#FFC48D',
            50: '#FFD1A6',
            30: '#FFE9D3',
            20: '#FFF2E5',
            10: '#FFF9F2',
          },
          sad: {
            100: '#6A4FBF',
            85: '#876FD0',
            70: '#A38FDF',
            60: '#B6A5E7',
            50: '#C9BCF0',
            30: '#E2D9F7',
            20: '#EEE8FB',
            10: '#F7F3FD',
          },
          relaxed: {
            100: '#28C76F',
            85: '#50D189',
            70: '#78DAA3',
            60: '#8EE0B1',
            50: '#A4E6C0',
            30: '#CCF2DD',
            20: '#E0F7EB',
            10: '#F2FBF6',
          },
          angry: {
            100: '#E63946',
            85: '#EB5F68',
            70: '#F0858A',
            60: '#F49DA0',
            50: '#F7B5B7',
            30: '#FBDADC',
            20: '#FDEAE9',
            10: '#FEF5F5',
          },
        },

        accent: {
          100: '#E650C5',
          85: '#EC73CD',
          70: '#F195D6',
          60: '#F4ACE0',
          50: '#F8CDEF',
          30: '#FBE3F7',
          20: '#FDEDFC',
          10: '#FFFBFF',
        },

        shadedOfGray: {
          100: '#000000',
          85: '#262626',
          70: '#4C4C4C',
          60: '#666666',
          50: '#808080',
          30: '#B2B2B2',
          20: '#CCCCCC',
          10: '#E6E6E6',
        },

        success: {
          'dark': '#00A300',
          'moderate': '#42E642',
          'lighten': '#8FFF8F',
        },

        warning: {
          'dark': '#A38200',
          'moderate': '#E6C042',
          'lighten': '#FFE58F',
        },

        error: {
          'dark': '#A30000',
          'moderate': '#E64242',
          'lighten': '#FF8F8F',
        },

        white: '#FFFFFF',
      },

      fontFamily: {
        'primary': ['poppins'],
        'secondaryAndButton': ['aeonik'],
      },

      fontSize: {
        'hero': ['4rem', {
          lineHeight: '6rem',
        }],

        'title': ['2.667rem', {
          lineHeight: '4rem',
        }],

        'subtitle': ['2rem', {
          lineHeight: '3rem',
        }],

        'paragraph': ['1.5rem', {
          lineHeight: '2.25rem',
        }],

        'body': ['1.167rem', {
          lineHeight: '1.75rem',
        }],

        'small': ['1rem', {
          lineHeight: '1.5rem',
        }],

        'micro': ['0.833rem', {
          lineHeight: '1.25rem',
        }],

        'smallBtn': ['1.333rem', {
          lineHeight: '2rem',
        }],

        'mediumBtn': ['1.5rem', {
          lineHeight: '2.25rem',
        }],

        'largeBtn': ['1.667rem', {
          lineHeight: '2.5rem',
        }],
      },
    },
  },
}