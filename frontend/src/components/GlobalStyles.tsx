export function GlobalStyles() {
  return (
    <style>{`
      :root {
        --border: 214.3 31.8% 83.9%;
        --input: 214.3 31.8% 83.9%;
        --ring: 222.2 84% 4.9%;
        --background: 0 0% 100%;
        --foreground: 222.2 84% 4.9%;
        --primary: 222.2 47.4% 11.2%;
        --primary-foreground: 210 40% 98%;
        --secondary: 210 40% 96%;
        --secondary-foreground: 222.2 47.4% 11.2%;
        --destructive: 0 84.2% 60.2%;
        --destructive-foreground: 210 40% 98%;
        --muted: 210 40% 96%;
        --muted-foreground: 215.4 16.3% 46.9%;
        --accent: 210 40% 96%;
        --accent-foreground: 222.2 47.4% 11.2%;
        --popover: 0 0% 100%;
        --popover-foreground: 222.2 84% 4.9%;
        --card: 0 0% 100%;
        --card-foreground: 222.2 84% 4.9%;
      }

      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }

      body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
          'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
          sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
      }

      @keyframes bounce {
        0%, 100% {
          transform: translateY(0);
          opacity: 1;
        }
        50% {
          transform: translateY(-10px);
          opacity: 0.7;
        }
      }

      .animate-bounce {
        animation: bounce 1s infinite;
      }

      .delay-100 {
        animation-delay: 0.1s;
      }

      .delay-200 {
        animation-delay: 0.2s;
      }
    `}</style>
  )
}
